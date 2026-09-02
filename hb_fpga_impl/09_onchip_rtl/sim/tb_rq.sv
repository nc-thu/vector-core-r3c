// ============================================================================
// tb_rq.sv — requant 二代（门 1）位精确对拍台
//   相位 A: s=8，rq_v1(神谕) vs {rq_v2 XW27/XW32/T_MAX39, rq_ms SHARE2/4} 逐位
//   相位 B: s∈[8,47]，rq_v1 vs rq_v2(T_MAX=39)
//   相位 C: s=8，rq_v1 vs rq_m6 → 偏差统计 + rq_m6_dump.txt（数值决策归用户）
// 用法: iverilog -g2012 -o tb_rq.vvp tb_rq.sv ../rtl/rq_v1.sv ../rtl/rq_v2.sv \
//                    ../rtl/rq_ms.sv ../rtl/rq_m6.sv && vvp tb_rq.vvp
// ============================================================================
`timescale 1ns/1ps
module tb_rq;
  localparam XW = 27;

  reg clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;

  // ---------------- DUT: v1 神谕 ----------------
  reg               v_in;
  reg  signed [31:0] v1_x;
  reg  signed [15:0] v1_m;
  reg  [7:0]         v1_s;
  wire               v1_ov;
  wire signed [7:0]  v1_y;
  rq_v1 u_v1 (.clk(clk), .rst_n(rst_n), .in_vld(v_in), .x(v1_x), .m(v1_m),
              .s(v1_s), .out_vld(v1_ov), .y(v1_y));

  // ---------------- DUT: rq_v2 三个配置 ----------------
  reg               a_in;
  reg  signed [XW-1:0] a_x;
  wire              a27_ov, a32_ov, g27_ov;
  wire signed [7:0] a27_y, a32_y, g27_y;
  rq_v2 #(.XW(27), .T_MAX(0))  u_a27 (.clk(clk), .rst_n(rst_n), .in_vld(a_in),
      .x(a_x), .m(v1_m), .s(8'd8), .out_vld(a27_ov), .y(a27_y));
  rq_v2 #(.XW(32), .T_MAX(0))  u_a32 (.clk(clk), .rst_n(rst_n), .in_vld(a_in),
      .x({{5{a_x[26]}}, a_x}), .m(v1_m), .s(8'd8), .out_vld(a32_ov), .y(a32_y));
  rq_v2 #(.XW(27), .T_MAX(39)) u_g27 (.clk(clk), .rst_n(rst_n), .in_vld(a_in),
      .x(a_x), .m(v1_m), .s(v1_s), .out_vld(g27_ov), .y(g27_y));

  // ---------------- DUT: rq_ms SHARE=2/4 ----------------
  reg  [1:0] ms2_in;
  reg  [2*XW-1:0] ms2_x;
  wire [1:0] ms2_ov;
  wire [15:0] ms2_y;
  rq_ms #(.SHARE(2), .XW(XW), .T_MAX(0)) u_ms2 (.clk(clk), .rst_n(rst_n),
      .in_vld(ms2_in), .x_bus(ms2_x), .m(v1_m), .s(8'd8),
      .out_vld(ms2_ov), .y_bus(ms2_y));

  reg  [3:0] ms4_in;
  reg  [4*XW-1:0] ms4_x;
  wire [3:0] ms4_ov;
  wire [31:0] ms4_y;
  rq_ms #(.SHARE(4), .XW(XW), .T_MAX(0)) u_ms4 (.clk(clk), .rst_n(rst_n),
      .in_vld(ms4_in), .x_bus(ms4_x), .m(v1_m), .s(8'd8),
      .out_vld(ms4_ov), .y_bus(ms4_y));

  // ---------------- DUT: rq_m6 ----------------
  reg               c_in;
  reg  signed [5:0]  c_m6;
  reg  signed [4:0]  c_t6;
  wire               c_ov;
  wire signed [7:0]  c_y;
  rq_m6 #(.XW(XW)) u_m6 (.clk(clk), .rst_n(rst_n), .in_vld(c_in), .x(a_x),
      .m6(c_m6), .t6(c_t6), .out_vld(c_ov), .y(c_y));

  // ---------------- 存储 ----------------
  reg [31:0] ctrl [0:2];
  reg [26:0] xm  [0:99999];
  reg [15:0] mm  [0:99999];
  reg [7:0]  sm  [0:99999];
  reg [31:0] m6m [0:99999];
  reg [31:0] t6m [0:99999];
  reg signed [7:0] ey  [0:99999];   // v1 神谕输出队列
  reg signed [7:0] q2  [0:99999];
  reg signed [7:0] q4  [0:99999];

  integer nA, nB, nC, i, j;
  integer err_a, err_b, eyn, q2n, q4n;
  integer cnt_c, diff_c, sum_abs, max_abs, dab, fd;
  integer skip_b, cmp_b, fdb;
  // 相位 C 打印用的 2 拍对齐影子（输出对应 i-2 的输入）
  reg signed [31:0] xp1, xp2;
  reg signed [15:0] mp1, mp2;
  reg signed [5:0]  m6p1, m6p2;
  reg signed [4:0]  t6p1, t6p2;

  task check_flush(input integer nflush);   // 冲刷期输出采集/对拍（相位 A 口径）
    integer k;
    begin
      for (k = 0; k < nflush; k = k + 1) begin
        @(negedge clk);
        step_a();
      end
    end
  endtask

  task step_a;   // 单拍：v1 与 a27/a32/g27 同拍比对；采集各队列
    begin
      if (v1_ov) begin
        ey[eyn] = v1_y; eyn = eyn + 1;
        if (a27_ov !== 1'b1 || a32_ov !== 1'b1 || g27_ov !== 1'b1) begin
          err_a = err_a + 1;
          if (err_a < 20) $display("[A-FAIL vld] x=%0d m=%0d", v1_x, v1_m);
        end else if (a27_y !== v1_y || a32_y !== v1_y || g27_y !== v1_y) begin
          err_a = err_a + 1;
          if (err_a < 20)
            $display("[A-FAIL] x=%0d m=%0d s=%0d v1=%0d a27=%0d a32=%0d g27=%0d",
                      v1_x, v1_m, v1_s, v1_y, a27_y, a32_y, g27_y);
        end
      end
      if (ms2_ov[0]) begin q2[q2n] = ms2_y[7:0];  q2n = q2n + 1; end
      if (ms2_ov[1]) begin q2[q2n] = ms2_y[15:8]; q2n = q2n + 1; end
      if (ms4_ov[0]) begin q4[q4n] = ms4_y[7:0];   q4n = q4n + 1; end
      if (ms4_ov[1]) begin q4[q4n] = ms4_y[15:8];  q4n = q4n + 1; end
      if (ms4_ov[2]) begin q4[q4n] = ms4_y[23:16]; q4n = q4n + 1; end
      if (ms4_ov[3]) begin q4[q4n] = ms4_y[31:24]; q4n = q4n + 1; end
    end
  endtask

  initial begin
    $readmemh("rq_ctrl.mem", ctrl);   // 避与 hif8 微架构 ctrl.mem 同名冲突
    nA = ctrl[0]; nB = ctrl[1]; nC = ctrl[2];
    err_a = 0; err_b = 0; eyn = 0; q2n = 0; q4n = 0;
    cnt_c = 0; diff_c = 0; sum_abs = 0; max_abs = 0;
    v_in = 0; a_in = 0; c_in = 0; ms2_in = 0; ms4_in = 0;
    v1_x = 0; v1_m = 0; v1_s = 0; a_x = 0; ms2_x = 0; ms4_x = 0;
    c_m6 = 0; c_t6 = 0;

    @(negedge clk); rst_n = 1'b1;
    repeat (3) @(negedge clk);

    // ================= 相位 A：s=8 =================
    $readmemh("xa.mem", xm);
    $readmemh("ma.mem", mm);
    for (i = 0; i < nA; i = i + 1) begin
      a_x  <= xm[i];
      v1_x <= {{5{xm[i][26]}}, xm[i]};
      v1_m <= mm[i];
      v1_s <= 8'd8;
      v_in <= 1'b1; a_in <= 1'b1;
      // ms 封装契约：列在 slot==c 拍上数据（slot 自由轮转，采样 slot_o 端口）
      ms2_in <= (2'b01 << u_ms2.slot_o);
      ms2_x[u_ms2.slot_o*XW +: XW] <= xm[i];
      ms4_in <= (4'b0001 << u_ms4.slot_o);
      ms4_x[u_ms4.slot_o*XW +: XW] <= xm[i];
      @(negedge clk);
      step_a();
    end
    v_in <= 0; a_in <= 0; ms2_in <= 0; ms4_in <= 0;
    check_flush(6);
    if (q2n != nA) begin err_a = err_a + 1; $display("[A-FAIL] ms2 输出数 %0d != %0d", q2n, nA); end
    if (q4n != nA) begin err_a = err_a + 1; $display("[A-FAIL] ms4 输出数 %0d != %0d", q4n, nA); end
    if (eyn  != nA) begin err_a = err_a + 1; $display("[A-FAIL] v1 输出数 %0d != %0d", eyn, nA); end
    for (j = 0; j < nA; j = j + 1) begin
      if (j < q2n && q2[j] !== ey[j]) begin
        err_a = err_a + 1;
        if (err_a < 20) $display("[A-FAIL ms2] j=%0d got=%0d exp=%0d", j, q2[j], ey[j]);
      end
      if (j < q4n && q4[j] !== ey[j]) begin
        err_a = err_a + 1;
        if (err_a < 20) $display("[A-FAIL ms4] j=%0d got=%0d exp=%0d", j, q4[j], ey[j]);
      end
    end
    $display("[PHASE-A] n=%0d err=%0d  (a27/a32/g27@8/ms2/ms4 vs v1)", nA, err_a);

    // ================= 相位 B：s∈[8,47] =================
    // v1 神谕的 s 在 T1 采当前拍（原版结构，静态 s 下无害）；g27 已改为移位量
    // 流水对齐。仅当 sm[j+1]==sm[j]（v1 实际用的 s 与向量自身 s 一致）时对拍。
    eyn = 0; skip_b = 0; cmp_b = 0;
    fdb = $fopen("pb_dbg.txt", "w");
    $readmemh("xb.mem", xm);
    $readmemh("mb.mem", mm);
    $readmemh("sb.mem", sm);
    for (i = 0; i < nB; i = i + 1) begin
      a_x  <= xm[i];
      v1_x <= {{5{xm[i][26]}}, xm[i]};
      v1_m <= mm[i]; v1_s <= sm[i];
      v_in <= 1'b1; a_in <= 1'b1;
      @(negedge clk);
      // 拍对齐：本拍（P_i 之后）v1 输出 = sat(P_{i-1}>>>s_i)，g27 = sat(P_{i-1}>>>s_{i-1})
      // （v1 在 T1 采当前拍 s —— 原版结构；g27 移位量已流水对齐）。故仅当
      // sm[i]==sm[i-1]（v1 实际移位量与向量自身一致）时可对拍，行归属向量 i-1。
      if (v1_ov) begin
        if (i >= 1 && sm[i] == sm[i-1]) begin
          cmp_b = cmp_b + 1;
          $fwrite(fdb, "%0d %0d %0d %0d %0d %0d\n", i-1, xm[i-1], mm[i-1], sm[i-1], v1_y, g27_y);
          if (g27_y !== v1_y) begin
            err_b = err_b + 1;
            if (err_b < 20)
              $display("[B-FAIL] j=%0d x=%0d m=%0d s_g27=%0d s_v1=%0d v1=%0d g27=%0d",
                        i-1, xm[i-1], mm[i-1], sm[i-1], sm[i], v1_y, g27_y);
          end
        end else skip_b = skip_b + 1;
      end
    end
    v_in <= 0; a_in <= 0;
    repeat (4) begin @(negedge clk);
      if (v1_ov && g27_ov) begin
        // 冲刷第 1 拍 = 向量 nB-1 的输出：v1 移位量 s 已保持 sm[nB-1]，两者恒可比
        cmp_b = cmp_b + 1;
        if (g27_y !== v1_y) begin
          err_b = err_b + 1;
          $display("[B-FAIL flush] v1=%0d g27=%0d", v1_y, g27_y);
        end
      end
    end
    $display("[PHASE-B] n=%0d cmp=%0d skip=%0d err=%0d  (g27 s∈[8,47] vs v1)",
             nB, cmp_b, skip_b, err_b);

    // ================= 相位 C：m6 偏差数据 =================
    $readmemh("xc.mem", xm);
    $readmemh("mc.mem", mm);
    $readmemh("m6c.mem", m6m);
    $readmemh("t6c.mem", t6m);
    fd = $fopen("rq_m6_dump.txt", "w");
    $fwrite(fd, "// x m m6 t6 y_v1 y_m6\n");
    for (i = 0; i < nC; i = i + 1) begin
      a_x  <= xm[i];
      v1_x <= {{5{xm[i][26]}}, xm[i]};
      v1_m <= mm[i]; v1_s <= 8'd8;
      c_m6 <= m6m[i][5:0]; c_t6 <= t6m[i][4:0];
      xp1 <= v1_x; mp1 <= v1_m; m6p1 <= c_m6; t6p1 <= c_t6;
      xp2 <= xp1;  mp2 <= mp1;  m6p2 <= m6p1;  t6p2 <= t6p1;
      v_in <= 1'b1; c_in <= 1'b1;
      @(negedge clk);
      if (v1_ov && c_ov) begin
        cnt_c = cnt_c + 1;
        if (c_y !== v1_y) begin
          dab = c_y - v1_y;
          if (dab < 0) dab = -dab;
          diff_c = diff_c + 1;
          sum_abs = sum_abs + dab;
          if (dab > max_abs) max_abs = dab;
          if (diff_c <= 5000)
            $fwrite(fd, "%0d %0d %0d %0d %0d %0d\n",
                    xp1, mp1, m6p1, t6p1, v1_y, c_y);   // 输出归属向量 i-1 → xp1
        end
      end
    end
    v_in <= 0; c_in <= 0;
    repeat (4) begin @(negedge clk);
      if (v1_ov && c_ov) begin
        cnt_c = cnt_c + 1;
        if (c_y !== v1_y) begin
          dab = c_y - v1_y; if (dab < 0) dab = -dab;
          diff_c = diff_c + 1; sum_abs = sum_abs + dab;
          if (dab > max_abs) max_abs = dab;
        end
      end
    end
    $fclose(fd); $fclose(fdb);
    $display("[PHASE-C] n=%0d diff=%0d (%.2f%%) mean_abs=%.4f max_abs=%0d",
             cnt_c, diff_c, 100.0*diff_c/cnt_c, (1.0*sum_abs)/cnt_c, max_abs);

    if (err_a == 0 && err_b == 0)
      $display("TB_RQ PASS (exact variants bit-identical to v1)");
    else
      $display("TB_RQ FAIL (err_a=%0d err_b=%0d)", err_a, err_b);
    $finish;
  end
endmodule
