// ============================================================================
// ae_hif8_dot16.sv — HiF8-native 16 乘积/拍 点积微架构（route C 判定实验②）
// ----------------------------------------------------------------------------
// 格式  : Ascend HiFloat8（arXiv 2409.16626）
//         S(1) + dot 前缀码(2~4b, 表 D∈{0..4}/DML) + Em 符号-数值指数(D b, 最高
//         幅值位隐含 1) + 锥化尾数(5-D b)。正常数 E∈[-15,15]；denormal 把 E 扩到
//         [-22,-16]（值 = 2^(M-23)，M∈[1,7]），共 38 binade。
//         特殊值处理：0x00/0x80 → 0（NaN 按零）；Inf(S 11 0111 1) 按数值
//         1.5×2^15 精确解码，不设特殊通路（微架构实验口径，报告有说明）。
// 结构  : 每 lane（= 输出驻留阵列一列的 16 个输出累加器之一）一条
//         解码 → 4×4 尾数整乘(隐含位在内, sig∈[8,15]) → 指数加 → 4 binade/桶
//         的整数桶累加（19 桶 × ACC_W，读-改-写、全程无中途舍入）。
//         末端共享一个串行单元：Horner 合并 19 桶 → 98b 定点整数（刻度 2^-50：W = Σ sig8·2^e7 = 64·2^44·v） → 左移规格化
//         → TA(半向上)舍入 → HiF8 编码（饱和到最大正常数 2^15）。
// 参数  : PD ∈{1,2,3} 流水深度；SHARE_B=1 时 B 广播共享解码（阵列列口径）；
//         K_MAX 决定 ACC_W（K_MAX=4096 时累加上界 4096×225×8 < 2^23）。
// 判据  : LUT/乘积 ≤60 可行 / =35 追平 packed INT8；Fmax ≥180 MHz；DSP 必须=0。
// ============================================================================
module ae_hif8_dot16 #(
    parameter PD      = 2,     // 1/2/3：输入寄存 →(解码)→(乘)→ 桶RMW 的切分
    parameter SHARE_B = 0,     // 1 = b_q 只用低 8 bit 广播给 16 lane
    parameter K_MAX   = 4096,
    parameter ACC_W   = 25,    // K_MAX=4096 精确上界 24 bit，留 1 bit 余量
    parameter LANES   = 16,
    parameter W_W     = 98     // 合并宽度 = ACC_W + 4*18 + 1（符号）
)(
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,    // idle 时脉冲：清桶，开始新 dot
    input  wire         i_valid,
    input  wire         i_last,   // 本 dot 最后一个 i_valid 拍
    input  wire [127:0] a_q,      // lane i 操作数 = a_q[8*i +: 8]
    input  wire [127:0] b_q,      // SHARE_B=1 时只用 b_q[7:0]
    output wire         busy,
    output reg          o_valid,
    output reg  [3:0]   o_lane,
    output reg  [7:0]   o_byte
);
    localparam NB = 19;           // e7∈[0,74] → e7>>2 ∈[0,18]

    // ------------------------------------------------------------------
    // 状态/数据寄存声明（先声明后使用）
    // ------------------------------------------------------------------
    localparam ST_IDLE = 3'd0, ST_RUN = 3'd1, ST_C0 = 3'd2, ST_C = 3'd3,
                ST_CA = 3'd4, ST_N = 3'd5, ST_E = 3'd6;
    reg [2:0] st;
    reg         v1, last1, v2, last2, v3, last3;
    // 桶 RMW 级有效位：与该级数据同拍（PD=1 用 v1 直通，PD=2 用 v2，PD=3 用 v3）
    wire        vf    = (PD == 1) ? v1 : (PD == 2) ? v2 : v3;
    wire        lastf = (PD == 1) ? last1 : (PD == 2) ? last2 : last3;
    reg [127:0] a1, b1;
    reg [20:0]  d2 [0:LANES-1];  // {sgn, sa4, ea6, sb4, eb6}
    reg [15:0]  d3 [0:LANES-1];  // {sig8, e7, sgn}
    reg signed [ACC_W-1:0] bfile [0:LANES-1][0:NB-1];
    reg [3:0] r_lane, r_ld_lane;
    reg [4:0] r_cb;
    reg signed [W_W-1:0] W;
    reg [W_W-1:0] A;
    reg [6:0] s_cnt;
    reg       r_sign;
    reg [2:0] r_k;
    reg signed [6:0] r_E;
    integer i, j;

    // ------------------------------------------------------------------
    // 解码：byte → {s(1), sig4(4, 0=零/NaN), eidx(6, =E+22)}
    // 位布局（msb=bit7=S）：D=4: dot=11,  Em=b4..b1, M=b0
    //                      D=3: dot=10,  Em=b4..b2, M=b1..b0
    //                      D=2: dot=01,  Em=b4..b3, M=b2..b0
    //                      D=1: dot=001, Em=b3,     M=b2..b0
    //                      D=0: dot=0001,M=b2..b0,  E=0
    //                      DML: dot=0000,M=b2..b0,  值=2^(M-23)
    // ------------------------------------------------------------------
    function automatic [10:0] hif8_dec(input [7:0] c);
        reg d4, d3f, d2f, d1, d0, dml, se;
        reg [4:0] mag;                 // DML 需 17..22（=23-M），4 bit 装不下
        reg [3:0] sig;
        reg [5:0] ex;
        begin
            d4  = (c[6:5] == 2'b11);
            d3f = (c[6:5] == 2'b10);
            d2f = (c[6:5] == 2'b01);
            d1  = (c[6:4] == 3'b001);
            d0  = (c[6:3] == 4'b0001);
            dml = (c[6:3] == 4'b0000);
            se  = 1'b0; mag = 5'd0; sig = {1'b1, c[2:0]};
            case (1'b1)
                d4:  begin mag = {2'b01, c[3:1]};     se = c[4]; sig = {1'b1, c[0], 2'b00}; end
                d3f: begin mag = {3'b001, c[3:2]};     se = c[4]; sig = {1'b1, c[1:0], 1'b0}; end
                d2f: begin mag = {4'b0001, c[3]};      se = c[4]; end
                d1:  begin mag = 5'd1; se = c[3]; end
                d0:  begin mag = 5'd0; se = 1'b0; end
                dml: begin se = 1'b1; mag = 5'd23 - {2'b00, c[2:0]}; sig = 4'd8; end
                default: ;
            endcase
            ex = se ? (6'd22 - {1'b0, mag}) : (6'd22 + {1'b0, mag});
            if (dml && c[2:0] == 3'd0) begin  // 0x00 零 / 0x80 NaN→0（桶索引钳到 0，加的是 ±0）
                sig = 4'd0; ex = 6'd0;
            end
            hif8_dec = {c[7], sig, ex};
        end
    endfunction

    // ------------------------------------------------------------------
    // 组合级连线
    // ------------------------------------------------------------------
    wire [10:0] dec_a [0:LANES-1];
    wire [10:0] dec_b [0:LANES-1];
    wire [3:0] m_sa [0:LANES-1];
    wire [3:0] m_sb [0:LANES-1];
    wire [5:0] m_ea [0:LANES-1];
    wire [5:0] m_eb [0:LANES-1];
    wire       m_g  [0:LANES-1];
    (* use_dsp = "no" *) wire [7:0] prod_s8 [0:LANES-1];  // sa*sb∈[0,225]，具名线网强制 LUT
    wire [6:0] prod_e7 [0:LANES-1];                       // (Ea+22)+(Eb+22)=E+44∈[0,74]
    wire       prod_g  [0:LANES-1];
    wire [7:0] f_s8 [0:LANES-1];
    wire [6:0] f_e7 [0:LANES-1];
    wire       f_g  [0:LANES-1];
    wire [4:0]  f_bidx [0:LANES-1];
    wire [11:0] f_vals [0:LANES-1];   // 符号化 (sig8<<r) ∈[-1800,1800]

    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : glane
            // 解码（PD>=2 时驱动 d2 寄存；PD=1 时直通乘法）
            assign dec_a[g] = hif8_dec(a1[8*g +: 8]);
            assign dec_b[g] = (SHARE_B != 0) ? hif8_dec(b1[7:0]) : hif8_dec(b1[8*g +: 8]);
            // 乘法源选择：PD==1 ? 解码级 : d2 寄存
            assign m_sa[g] = (PD == 1) ? dec_a[g][9:6] : d2[g][19:16];
            assign m_sb[g] = (PD == 1) ? dec_b[g][9:6] : d2[g][9:6];
            assign m_ea[g] = (PD == 1) ? dec_a[g][5:0] : d2[g][15:10];
            assign m_eb[g] = (PD == 1) ? dec_b[g][5:0] : d2[g][5:0];
            assign m_g[g]  = (PD == 1) ? (dec_a[g][10] ^ dec_b[g][10]) : d2[g][20];
            // 4×4 尾数整乘 + 指数加
            assign prod_s8[g] = m_sa[g] * m_sb[g];
            assign prod_e7[g] = m_ea[g] + m_eb[g];
            assign prod_g[g]  = m_g[g];
            // 桶 RMW 源：PD==3 ? d3 寄存 : 乘法组合级
            assign f_s8[g]   = (PD == 3) ? d3[g][15:8] : prod_s8[g];
            assign f_e7[g]   = (PD == 3) ? d3[g][7:1]  : prod_e7[g];
            assign f_g[g]    = (PD == 3) ? d3[g][0]    : prod_g[g];
            assign f_bidx[g] = f_e7[g][6:2];
            // 桶内幅值 = sig8<<r（桶 r 低两位），先拓宽到 12b 再取补 —— 直接对 11b
            // $signed 取负会溢出（1800 > 1023），且拼接 {s8,r} 是 s8*4+r 不是移位
            wire [10:0] f_mag = {3'b000, f_s8[g]} << f_e7[g][1:0];  // ∈[0,1800]
            wire [11:0] f_mag12 = {1'b0, f_mag};
            assign f_vals[g] = f_g[g] ? (~f_mag12 + 12'd1) : f_mag12;
        end
    endgenerate

    // ------------------------------------------------------------------
    // 流水有效位：s1 输入 →[PD>=2] s2 解码 →[PD>=3] s3 乘积 → 桶 RMW
    // ------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0;
            last1 <= 1'b0; last2 <= 1'b0; last3 <= 1'b0;
        end else begin
            v1    <= i_valid && (st == ST_RUN);
            last1 <= i_valid && i_last && (st == ST_RUN);
            v2    <= (PD >= 2) && v1;   last2 <= (PD >= 2) && last1;
            v3    <= (PD >= 3) && v2;   last3 <= (PD >= 3) && last2;
            a1 <= a_q; b1 <= b_q;
            for (i = 0; i < LANES; i = i + 1) begin
                d2[i] <= {dec_a[i][10] ^ dec_b[i][10], dec_a[i][9:6], dec_a[i][5:0],
                          dec_b[i][9:6], dec_b[i][5:0]};
                d3[i] <= {prod_s8[i], prod_e7[i], prod_g[i]};
            end
        end
    end

    // ------------------------------------------------------------------
    // 桶文件 RMW（同拍同址先读旧值再加；start 清桶优先）
    // ------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (start && st == ST_IDLE) begin
            for (i = 0; i < LANES; i = i + 1)
                for (j = 0; j < NB; j = j + 1)
                    bfile[i][j] <= {ACC_W{1'b0}};
        end else if (vf && st == ST_RUN) begin
            for (i = 0; i < LANES; i = i + 1)
                bfile[i][f_bidx[i]] <= bfile[i][f_bidx[i]] +
                    $signed({{(ACC_W-12){f_vals[i][11]}}, f_vals[i]});
        end
    end

    // ------------------------------------------------------------------
    // 末端合并/规格化/编码组合量
    // ------------------------------------------------------------------
    // 锥化表：k = 目标尾数位数（7 = 饱和/下溢哨兵，不走舍入路径）
    function automatic [2:0] k_taper(input signed [6:0] E);
        begin
            if (E >= 7'sd16 || E <= -7'sd23)   k_taper = 3'd7;
            else if (E <= -7'sd16)             k_taper = 3'd0;  // DML 纯 2 幂
            else if (E >= -7'sd3 && E <= 7'sd3) k_taper = 3'd3;
            else if (E >= -7'sd7 && E <= 7'sd7) k_taper = 3'd2;
            else                               k_taper = 3'd1;  // |E|∈[8,15]
        end
    endfunction

    // 编码：E∈[-22,15]。DML(E∈[-22,-16]) 编 M=E+23；正常数按 |E| 选 dot/Em/尾数宽
    function automatic [7:0] hif8_enc(input s, input signed [6:0] E, input [3:0] frac);
        reg [3:0] mag;
        reg [2:0] dm;
        begin
            if (E <= -7'sd16) begin
                dm = E[2:0] + 3'd7;                  // (23+E) mod 8，结果∈[1,7]
                hif8_enc = {s, 4'b0000, dm};
            end else begin
                mag = (E < 0) ? -E : E;
                if (mag == 4'd0)      hif8_enc = {s, 4'b0001, frac[2:0]};
                else if (mag == 4'd1) hif8_enc = {s, 3'b001, E[6], frac[2:0]};
                else if (mag <= 4'd3) hif8_enc = {s, 2'b01, E[6], mag[0], frac[2:0]};
                else if (mag <= 4'd7) hif8_enc = {s, 2'b10, E[6], mag[1:0], frac[1:0]};
                else                  hif8_enc = {s, 2'b11, E[6], mag[2:0], frac[0]};
            end
        end
    endfunction

    // Horner：W ← W*16 + acc[cb]（初值 acc18×16 → 终值 Σ acc_b×2^(4b)，刻度 2^-50（sig8 含 3 位小数尺度））
    wire signed [W_W-1:0] w_next = (W <<< 4) +
        $signed({{(W_W-ACC_W){bfile[r_lane][r_cb][ACC_W-1]}}, bfile[r_lane][r_cb]});
    // 装载 lane 的桶 18（直接符号扩展作 Horner 初值 —— 后续 18 步 ×16 已把它的
    // 系数推到 16^18，再预移 4 会到 16^19，桶 18 整体放大 16 倍）
    wire signed [W_W-1:0] b18_ld = $signed({{(W_W-ACC_W){bfile[r_ld_lane][NB-1][ACC_W-1]}},
                                             bfile[r_ld_lane][NB-1]});

    // 规格化舍入量（A 的 MSB 在 [W_W-1]，frac/k 按 r_k 取 MSB 下的位）
    wire [3:0] n_frac = (r_k == 3'd0) ? 4'd0 :
                        (r_k == 3'd1) ? {3'd0, A[W_W-2]} :
                        (r_k == 3'd2) ? {2'd0, A[W_W-2], A[W_W-3]} :
                                        {1'b0, A[W_W-2], A[W_W-3], A[W_W-4]};
    wire n_half = (r_k == 3'd0) ? A[W_W-2] : (r_k == 3'd1) ? A[W_W-3] :
                  (r_k == 3'd2) ? A[W_W-4] : A[W_W-5];
    wire [4:0] n_fsum = {1'b0, n_frac} + {4'd0, n_half};
    wire n_carry = (r_k == 3'd0) ? n_fsum[0] : (r_k == 3'd1) ? n_fsum[1] :
                   (r_k == 3'd2) ? n_fsum[2] : n_fsum[3];

    assign busy = (st != ST_IDLE);

    // ------------------------------------------------------------------
    // FSM：IDLE → RUN → C0(装桶18) → C(Horner 17..0) → CA(取绝对值)
    //      → N(左移规格化) → E(舍入+编码+输出) → 下一 lane / IDLE
    // ------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= ST_IDLE; o_valid <= 1'b0; o_lane <= 4'd0; o_byte <= 8'd0;
        end else begin
            o_valid <= 1'b0;
            case (st)
                ST_IDLE: if (start) st <= ST_RUN;
                ST_RUN: if (lastf) begin
                    r_lane    <= 4'd0;    // 经 C0 隔一拍，等最后一个 RMW 落桶
                    r_ld_lane <= 4'd0;
                    st <= ST_C0;
                end
                ST_C0: begin
                    W <= b18_ld;          // = acc18（Horner 初值）
                    r_cb <= NB-2;
                    st <= ST_C;
                end
                ST_C: begin
                    W <= w_next;
                    if (r_cb == 5'd0) st <= ST_CA;
                    else r_cb <= r_cb - 5'd1;
                end
                ST_CA: begin              // W 已是全和
                    A <= (W < 0) ? (-W) : W;
                    r_sign <= W[W_W-1];
                    s_cnt <= 7'd0;
                    st <= ST_N;
                end
                ST_N: begin
                    if (A[W_W-1]) begin
                        r_E <= $signed({1'b0, 7'd97 - s_cnt}) - 7'sd50;
                        r_k <= k_taper($signed({1'b0, 7'd97 - s_cnt}) - 7'sd50);
                        st <= ST_E;
                    end else if (s_cnt == 7'd97) begin
                        r_E <= -7'sd64;   // 零和标记
                        st <= ST_E;
                    end else begin
                        A <= A << 1;
                        s_cnt <= s_cnt + 7'd1;
                    end
                end
                ST_E: begin
                    o_lane <= r_lane;
                    o_valid <= 1'b1;
                    if (r_E == -7'sd64)
                        o_byte <= 8'h00;                                  // 零和
                    else if (r_E >= 7'sd15)                               // E=15 仅 1.0×2^15 可表示
                        o_byte <= {r_sign, 2'b11, 4'b0111, 1'b0};        // 饱和 2^15（含进位到 15/16）
                    else if (r_E <= -7'sd23)                             // 下溢
                        o_byte <= (s_cnt <= 7'd70) ? {r_sign, 4'b0000, 3'd001} : 8'h00;
                    else if (n_carry)
                        o_byte <= hif8_enc(r_sign, r_E + 7'sd1, 4'd0);   // 进位→1.0×2^(E+1)
                    else
                        o_byte <= hif8_enc(r_sign, r_E, n_frac);         // 含 DML(k=0)
                    // 下一 lane 或收尾
                    if (r_lane == 4'd15) st <= ST_IDLE;
                    else begin
                        r_lane    <= r_lane + 4'd1;
                        r_ld_lane <= r_lane + 4'd1;
                        r_cb      <= NB-2;
                        st <= ST_C0;
                    end
                end
                default: st <= ST_IDLE;
            endcase
        end
    end
endmodule
