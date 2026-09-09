// ============================================================================
// ae_hif8_dot16_m2.sv — HiF8 dot 二代尝试 A（门 2）：桶文件改分布式 LUTRAM
// ----------------------------------------------------------------------------
// 与一代 ae_hif8_dot16 的唯一结构差异：
//   一代 bfile = 19×ACC_b 寄存器堆 + 19:1 读选择 + 全字并行 start 清零 +
//               每字写使能门控（实测每路 ~402-460 LUT，桶文件占大头）。
//   二代 M2-A  = 每 lane 一块 32 深 × ACC_W 分布式 RAM（RAM32X1S 推断），
//               单端口同址「异步读-改-写」：
//                 RUN  段：addr=f_bidx，读旧值+val 写回（RMW）；
//                 合并段：addr=合并桶号，读旧值进 Horner，同沿写 0（消费清零）。
//               start 并行清零整个删除 —— 每个 dot 的 19 桶在 Horner 合并时
//               逐桶消费清零（0 额外拍数），全和与一代逐位一致（TB 对拍）。
//   上电零初始化（initial 常量 0 → LUTRAM INIT）；此后全靠消费清零维持零态。
// 其余（解码 / 4×4 尾数乘 / 指数加 / Horner 合并 / 规格化 / TA 舍入 / 编码）
// 与一代逐行相同。数值口径：与一代位精确等价。
// 判据：LUT/积 ≤60 可行；Fmax ≥180 MHz（OOC 4ns 目标）；DSP=0。
// ============================================================================
module ae_hif8_dot16_m2 #(
    parameter PD      = 2,
    parameter SHARE_B = 0,
    parameter K_MAX   = 4096,
    parameter ACC_W   = 25,
    parameter LANES   = 16,
    parameter W_W     = 98,
    parameter NB      = 19,
    parameter DEPTH   = 32          // LUTRAM 深度（RAM32X1S 最小实用深度）
)(
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,      // idle 时脉冲：开始新 dot（不再清桶 —— 桶恒为消费后零态）
    input  wire         i_valid,
    input  wire         i_last,
    input  wire [127:0] a_q,
    input  wire [127:0] b_q,
    output wire         busy,
    output reg          o_valid,
    output reg  [3:0]   o_lane,
    output reg  [7:0]   o_byte
);
    localparam ST_IDLE = 3'd0, ST_RUN = 3'd1, ST_C0 = 3'd2, ST_C = 3'd3,
                ST_CA = 3'd4, ST_N = 3'd5, ST_E = 3'd6;
    reg [2:0] st;
    reg         v1, last1, v2, last2, v3, last3;
    wire        vf    = (PD == 1) ? v1 : (PD == 2) ? v2 : v3;
    wire        lastf = (PD == 1) ? last1 : (PD == 2) ? last2 : last3;
    reg [127:0] a1, b1;
    reg [20:0]  d2 [0:LANES-1];
    reg [15:0]  d3 [0:LANES-1];
    reg [3:0] r_lane;
    reg [4:0] r_cb;
    reg signed [W_W-1:0] W;
    reg [W_W-1:0] A;
    reg [6:0] s_cnt;
    reg       r_sign;
    reg [2:0] r_k;
    reg signed [6:0] r_E;
    integer i;

    // ------------------------------------------------------------------
    // 解码：byte → {s(1), sig4(4, 0=零/NaN), eidx(6, =E+22)}（与一代相同）
    // ------------------------------------------------------------------
    function automatic [10:0] hif8_dec(input [7:0] c);
        reg d4, d3f, d2f, d1, d0, dml, se;
        reg [4:0] mag;
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
            if (dml && c[2:0] == 3'd0) begin
                sig = 4'd0; ex = 6'd0;
            end
            hif8_dec = {c[7], sig, ex};
        end
    endfunction

    // ------------------------------------------------------------------
    // 组合级连线（与一代相同）
    // ------------------------------------------------------------------
    wire [10:0] dec_a [0:LANES-1];
    wire [10:0] dec_b [0:LANES-1];
    wire [3:0] m_sa [0:LANES-1];
    wire [3:0] m_sb [0:LANES-1];
    wire [5:0] m_ea [0:LANES-1];
    wire [5:0] m_eb [0:LANES-1];
    wire       m_g  [0:LANES-1];
    (* use_dsp = "no" *) wire [7:0] prod_s8 [0:LANES-1];
    wire [6:0] prod_e7 [0:LANES-1];
    wire       prod_g  [0:LANES-1];
    wire [7:0] f_s8 [0:LANES-1];
    wire [6:0] f_e7 [0:LANES-1];
    wire       f_g  [0:LANES-1];
    wire [4:0]  f_bidx [0:LANES-1];
    wire [11:0] f_vals [0:LANES-1];

    // ------------------------------------------------------------------
    // 桶 LUTRAM（每 lane 一块）：单写口 + 异步读，同址读-改-写
    // ------------------------------------------------------------------
    wire run_op = vf && (st == ST_RUN);                       // RMW 写
    wire mrg_st = (st == ST_C0) || (st == ST_C);              // 合并消费阶段
    wire [4:0] mrg_ad = (st == ST_C0) ? NB-1 : r_cb;          // 合并访问地址
    wire signed [ACC_W-1:0] brd [0:LANES-1];                  // 各 lane 异步读数据

    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : glane
            (* ram_style = "distributed" *) reg signed [ACC_W-1:0] mem [0:DEPTH-1];
            wire lane_mrg = mrg_st && (r_lane == g);
            wire [4:0] addr = run_op ? f_bidx[g] : mrg_ad;
            assign brd[g] = mem[addr];
            integer ii;
            initial for (ii = 0; ii < DEPTH; ii = ii + 1) mem[ii] = {ACC_W{1'b0}};
            always @(posedge clk) begin
                if (run_op)      mem[addr] <= brd[g] +
                    $signed({{(ACC_W-12){f_vals[g][11]}}, f_vals[g]});
                else if (lane_mrg) mem[addr] <= {ACC_W{1'b0}};   // 消费清零
            end
        end
    endgenerate

    generate
        for (g = 0; g < LANES; g = g + 1) begin : glane_dp
            assign dec_a[g] = hif8_dec(a1[8*g +: 8]);
            assign dec_b[g] = (SHARE_B != 0) ? hif8_dec(b1[7:0]) : hif8_dec(b1[8*g +: 8]);
            assign m_sa[g] = (PD == 1) ? dec_a[g][9:6] : d2[g][19:16];
            assign m_sb[g] = (PD == 1) ? dec_b[g][9:6] : d2[g][9:6];
            assign m_ea[g] = (PD == 1) ? dec_a[g][5:0] : d2[g][15:10];
            assign m_eb[g] = (PD == 1) ? dec_b[g][5:0] : d2[g][5:0];
            assign m_g[g]  = (PD == 1) ? (dec_a[g][10] ^ dec_b[g][10]) : d2[g][20];
            assign prod_s8[g] = m_sa[g] * m_sb[g];
            assign prod_e7[g] = m_ea[g] + m_eb[g];
            assign prod_g[g]  = m_g[g];
            assign f_s8[g]   = (PD == 3) ? d3[g][15:8] : prod_s8[g];
            assign f_e7[g]   = (PD == 3) ? d3[g][7:1]  : prod_e7[g];
            assign f_g[g]    = (PD == 3) ? d3[g][0]    : prod_g[g];
            assign f_bidx[g] = f_e7[g][6:2];
            wire [10:0] f_mag = {3'b000, f_s8[g]} << f_e7[g][1:0];
            wire [11:0] f_mag12 = {1'b0, f_mag};
            assign f_vals[g] = f_g[g] ? (~f_mag12 + 12'd1) : f_mag12;
        end
    endgenerate

    // ------------------------------------------------------------------
    // 流水有效位（与一代相同）
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
    // 末端合并/规格化/编码（与一代相同，读改 brd[r_lane]）
    // ------------------------------------------------------------------
    function automatic [2:0] k_taper(input signed [6:0] E);
        begin
            if (E >= 7'sd16 || E <= -7'sd23)   k_taper = 3'd7;
            else if (E <= -7'sd16)             k_taper = 3'd0;
            else if (E >= -7'sd3 && E <= 7'sd3) k_taper = 3'd3;
            else if (E >= -7'sd7 && E <= 7'sd7) k_taper = 3'd2;
            else                               k_taper = 3'd1;
        end
    endfunction

    function automatic [7:0] hif8_enc(input s, input signed [6:0] E, input [3:0] frac);
        reg [3:0] mag;
        reg [2:0] dm;
        begin
            if (E <= -7'sd16) begin
                dm = E[2:0] + 3'd7;
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

    // 合并读数：RUN 段后仅 r_lane 的读口被合并单元使用
    wire signed [ACC_W-1:0] mrd = brd[r_lane];
    wire signed [W_W-1:0] w_next = (W <<< 4) +
        $signed({{(W_W-ACC_W){mrd[ACC_W-1]}}, mrd});
    wire signed [W_W-1:0] b18_ld = $signed({{(W_W-ACC_W){mrd[ACC_W-1]}}, mrd});

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
    // FSM：IDLE → RUN → C0(读+清桶18,装初值) → C(Horner 17..0, 逐桶读+清)
    //      → CA(绝对值) → N(规格化) → E(舍入+编码+输出) → 下一 lane / IDLE
    // ------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= ST_IDLE; o_valid <= 1'b0; o_lane <= 4'd0; o_byte <= 8'd0;
        end else begin
            o_valid <= 1'b0;
            case (st)
                ST_IDLE: if (start) st <= ST_RUN;
                ST_RUN: if (lastf) begin
                    r_lane <= 4'd0;
                    st <= ST_C0;
                end
                ST_C0: begin
                    W <= b18_ld;
                    r_cb <= NB-2;
                    st <= ST_C;
                end
                ST_C: begin
                    W <= w_next;
                    if (r_cb == 5'd0) st <= ST_CA;
                    else r_cb <= r_cb - 5'd1;
                end
                ST_CA: begin
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
                        r_E <= -7'sd64;
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
                        o_byte <= 8'h00;
                    else if (r_E >= 7'sd15)
                        o_byte <= {r_sign, 2'b11, 4'b0111, 1'b0};
                    else if (r_E <= -7'sd23)
                        o_byte <= (s_cnt <= 7'd70) ? {r_sign, 4'b0000, 3'd001} : 8'h00;
                    else if (n_carry)
                        o_byte <= hif8_enc(r_sign, r_E + 7'sd1, 4'd0);
                    else
                        o_byte <= hif8_enc(r_sign, r_E, n_frac);
                    if (r_lane == 4'd15) st <= ST_IDLE;
                    else begin
                        r_lane <= r_lane + 4'd1;
                        r_cb   <= NB-2;
                        st <= ST_C0;
                    end
                end
                default: st <= ST_IDLE;
            endcase
        end
    end
endmodule
