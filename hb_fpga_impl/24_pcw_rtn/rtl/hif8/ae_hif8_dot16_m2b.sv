// ============================================================================
// ae_hif8_dot16_m2b.sv — HiF8 dot 二代尝试 B（门 2）：lane 对分时复用（II=2）
// ----------------------------------------------------------------------------
// 在 M2-A（LUTRAM 桶文件）之上，把「解码 → 4×4 乘 → 移位/取补 → RMW 加法」
// 数据通路按 lane 对 (2k, 2k+1) 分时复用：奇偶相位各处理半数 lane。
//   每对 1 个解码器 + 1 个乘法器 + 1 条 val/idx 通路 + 1 个桶加法器，
//   LUTRAM 桶文件仍每 lane 一块（桶状态无法共享 —— 每周期每 lane 各自 RMW）。
// 外部契约变化：输入向量每 2 拍收一个（i_valid 只在 phase=0 拍采样；phase=1 拍
// 忽略），K 拍喂入 → 2K 拍。数值与一代/M2-A 位精确一致（TB 对拍 expect.mem）。
// 判据：LUT/积 ≤60 可行；Fmax ≥180 MHz；DSP=0。II=2 如实报告。
// ============================================================================
module ae_hif8_dot16_m2b #(
    parameter ACC_W   = 25,
    parameter LANES   = 16,
    parameter W_W     = 98,
    parameter NB      = 19,
    parameter DEPTH   = 32
)(
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire         i_valid,    // 仅 phase=0 拍采样（II=2 契约）
    input  wire         i_last,
    input  wire [127:0] a_q,
    input  wire [127:0] b_q,        // 仅用 b_q[7:0]（SHARE_B=1 口径）
    output wire         busy,
    output reg          o_valid,
    output reg  [3:0]   o_lane,
    output reg  [7:0]   o_byte
);
    localparam PAIRS = LANES/2;
    localparam ST_IDLE = 3'd0, ST_RUN = 3'd1, ST_C0 = 3'd2, ST_C = 3'd3,
                ST_CA = 3'd4, ST_N = 3'd5, ST_E = 3'd6;
    reg [2:0] st;
    reg        phase;                    // 0=偶 lane 组 {2k}, 1=奇 lane 组 {2k+1}
    reg        a_vld, last_a;            // 输入向量采样（phase=0）→ 第 1 解码拍
    // 其余流水有效位在下方控制块内声明（a_vld_d/hv2_vld/...）
    reg [127:0] a1;
    reg [7:0]   b1;
    reg [3:0] r_lane;
    reg [4:0] r_cb;
    reg signed [W_W-1:0] W;
    reg [W_W-1:0] A;
    reg [6:0] s_cnt;
    reg       r_sign;
    reg [2:0] r_k;
    reg signed [6:0] r_E;

    // ------------------------------------------------------------------
    // 解码（与一代相同的函数）
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
    // 每对一条共享数据通路（PAIRS = LANES/2 条）
    // 流水： a1(phase 保持) →[解码, 组=phase]→ d2h →[乘, 组=~上一拍? 见下]→ RMW
    //   设 t 拍 phase=0：解码偶组 → t+1 拍 d2h=偶组（phase=1 拍），
    //   乘法在 t+1 拍算偶组 → RMW 偶组在 t+1 拍末；t+1 拍解码奇组 → t+2 拍 RMW 奇组。
    //   即 RMW 的 lane 组 = 本拍 phase（乘法源 d2h 为上一拍的组）。
    // ------------------------------------------------------------------
    wire [10:0] dec_a [0:PAIRS-1];
    wire [10:0] dec_b;
    wire [7:0]  a_byte [0:PAIRS-1];
    reg  [20:0] d2h [0:PAIRS-1];        // {sgn, sa4, ea6, sb4, eb6}
    wire [3:0] m_sa [0:PAIRS-1], m_sb [0:PAIRS-1];
    wire [5:0] m_ea [0:PAIRS-1], m_eb [0:PAIRS-1];
    wire       m_g  [0:PAIRS-1];
    (* use_dsp = "no" *) wire [7:0] prod_s8 [0:PAIRS-1];
    wire [6:0] prod_e7 [0:PAIRS-1];
    wire       prod_g  [0:PAIRS-1];
    wire [7:0]  f_s8 [0:PAIRS-1];
    wire [6:0]  f_e7 [0:PAIRS-1];
    wire        f_g  [0:PAIRS-1];
    wire [4:0]  f_bidx [0:PAIRS-1];
    wire [11:0] f_vals [0:PAIRS-1];

    assign dec_b = hif8_dec(b1);        // B 广播：全阵列共享一个解码（SHARE_B=1）
    genvar g;
    generate
        for (g = 0; g < PAIRS; g = g + 1) begin : gpair
            assign a_byte[g] = a1[8*(2*g+phase) +: 8];        // 本拍解码的组 = phase
            assign dec_a[g] = hif8_dec(a_byte[g]);
            // 乘法源 = d2h（上一拍解码的组，组号 = ~phase... 实际就是存着的那份）
            assign m_sa[g] = d2h[g][19:16];
            assign m_sb[g] = d2h[g][9:6];
            assign m_ea[g] = d2h[g][15:10];
            assign m_eb[g] = d2h[g][5:0];
            assign m_g[g]  = d2h[g][20];
            assign prod_s8[g] = m_sa[g] * m_sb[g];
            assign prod_e7[g] = m_ea[g] + m_eb[g];
            assign prod_g[g]  = m_g[g];
            assign f_s8[g]   = prod_s8[g];
            assign f_e7[g]   = prod_e7[g];
            assign f_g[g]    = prod_g[g];
            assign f_bidx[g] = f_e7[g][6:2];
            wire [10:0] f_mag = {3'b000, f_s8[g]} << f_e7[g][1:0];
            wire [11:0] f_mag12 = {1'b0, f_mag};
            assign f_vals[g] = f_g[g] ? (~f_mag12 + 12'd1) : f_mag12;
        end
    endgenerate

    // ------------------------------------------------------------------
    // 输入采样 / 流水有效位
    // 契约：向量在 phase=0 拍采样并保持 2 拍；解码拍 = 保持的第 1/2 拍（奇组、
    // 偶组各一拍，逐拍流水）；RMW 在解码拍 +1（乘法同拍组合完成）。
    //   a_vld(c)=1  → c 是某向量的第 1 个解码拍（奇组）
    //   a_vld_d(c)=1→ c 是其第 2 个解码拍（偶组）
    //   hv_vld = 本拍是有效解码拍；hv2_vld = 本拍是有效 RMW 拍（源=上一拍解码）
    //   RMW 写入组 = 上一拍 phase 选中的组 = ~phase（RAM 写门控用）
    // ------------------------------------------------------------------
    reg  a_vld_d, last_a_d;
    reg  hv2_vld, hv2_last;
    wire in_sample = i_valid && (st == ST_RUN) && (phase == 1'b0);
    wire hv_vld  = (st == ST_RUN) && (a_vld | a_vld_d);
    wire hv_last = a_vld_d & last_a_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase <= 1'b0; a_vld <= 1'b0; last_a <= 1'b0;
            a_vld_d <= 1'b0; last_a_d <= 1'b0;
            hv2_vld <= 1'b0; hv2_last <= 1'b0;
            a1 <= '0; b1 <= '0;
        end else begin
            phase <= ~phase;
            if (st == ST_RUN) begin
                a1 <= (in_sample) ? a_q : a1;
                b1 <= (in_sample) ? b_q[7:0] : b1;
                a_vld  <= in_sample;
                last_a <= in_sample && i_last;
            end else begin
                a_vld <= 1'b0; last_a <= 1'b0;
            end
            a_vld_d  <= a_vld;   last_a_d <= last_a;
            hv2_vld  <= hv_vld;  hv2_last <= hv_last;
        end
    end
    // （d2h 寄存单独写，避免与有效位块混杂）
    always_ff @(posedge clk) begin
        integer k;
        for (k = 0; k < PAIRS; k = k + 1)
            d2h[k] <= {dec_a[k][10] ^ dec_b[10], dec_a[k][9:6], dec_a[k][5:0],
                       dec_b[9:6], dec_b[5:0]};
    end

    // ------------------------------------------------------------------
    // 桶 LUTRAM（每 lane 一块）+ 分相 RMW / 合并消费清零（同 M2-A）
    // ------------------------------------------------------------------
    wire run_op = hv2_vld && (st == ST_RUN);        // 本拍 RMW（源=上一拍解码组）
    wire mrg_st = (st == ST_C0) || (st == ST_C);
    wire [4:0] mrg_ad = (st == ST_C0) ? NB-1 : r_cb;
    wire signed [ACC_W-1:0] brd [0:LANES-1];

    generate
        for (g = 0; g < LANES; g = g + 1) begin : glm
            (* ram_style = "distributed" *) reg signed [ACC_W-1:0] mem [0:DEPTH-1];
            localparam int PK  = g/2;               // 所属对
            localparam bit ODD = (g % 2) == 1;      // 奇偶组
            wire in_run_grp = (ODD != phase);       // RMW 组 = 上一拍 phase 选中组
            wire [4:0] raddr = run_op ? f_bidx[PK] : mrg_ad;
            wire lane_mrg = mrg_st && (r_lane == g);
            assign brd[g] = mem[raddr];
            integer ii;
            initial for (ii = 0; ii < DEPTH; ii = ii + 1) mem[ii] = {ACC_W{1'b0}};
            always @(posedge clk) begin
                if (run_op && in_run_grp)
                    mem[raddr] <= brd[g] +
                        $signed({{(ACC_W-12){f_vals[PK][11]}}, f_vals[PK]});
                else if (lane_mrg)
                    mem[raddr] <= {ACC_W{1'b0}};
            end
        end
    endgenerate

    // ------------------------------------------------------------------
    // 末端合并/规格化/编码（同一代/M2-A）
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
    // FSM（同 M2-A；RUN 结束等乘积排空 2 拍 —— last_f 链自然覆盖）
    // ------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= ST_IDLE; o_valid <= 1'b0; o_lane <= 4'd0; o_byte <= 8'd0;
        end else begin
            o_valid <= 1'b0;
            case (st)
                ST_IDLE: if (start) st <= ST_RUN;
                ST_RUN: if (hv2_last) begin
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
