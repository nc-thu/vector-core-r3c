// ae_exp_lut.sv — exp2 查找表（softmax 用）
// 输入 d = max - s（无符号 0..128，int8 分数差），输出 EXP[d] = round(2^(-d/16) * 4096)。
// 129 项 x 13 bit，分布式 ROM，表文件 exp2_lut.mem 由 sim/gen_vectors.py 生成。
`ifndef AE_EXP_LUT_SV
`define AE_EXP_LUT_SV
module ae_exp_lut (
  input  logic        [7:0] d,
  output logic signed [12:0] e    // 16..4096，恒正
);
  (* rom_style = "distributed" *) logic [12:0] tbl [0:128];
  initial $readmemh("exp2_lut.mem", tbl);

  logic [7:0] di;
  assign di = (d > 8'd128) ? 8'd128 : d;
  assign e  = tbl[di];
endmodule
`endif
