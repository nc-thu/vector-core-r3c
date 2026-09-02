// ae_ctx_ram.sv — CTX 主存：简单双端口（A 口只读 128b / B 口只写 128b + 逐字节 we）
// 16 个 lane bank 共享同一字地址 -> 一条 128b 字（lane*8 为字节位）。
// SDP 结构与 URAM288（1 写口 + 1 读口，72b，带 BWE）精确匹配：
//   128b -> 2 列 72b（列内 9 字节用 8 个，2 列共 16 字节）；
//   深度级联由综合器展开（ram_style=ultra）。前提（由 ae_core 保证）：
//   A/B 口永远分属不同引擎状态或不同地址（softmax P3 = B 写 P[j] 同时 A 预读 S[j+1]）。
`ifndef AE_CTX_RAM_SV
`define AE_CTX_RAM_SV
module ae_ctx_ram #(
  parameter int WORDS = 131072,
  parameter RAM_STYLE = "ultra"
)(
  input  logic clk,
  // A 口：读广播（128b = 16 lane 字节）
  input  logic [$clog2(WORDS)-1:0] raddr,
  output logic [127:0]             rdata,
  // B 口：写（逐字节使能）
  input  logic [15:0]              we_byte,
  input  logic [$clog2(WORDS)-1:0] waddr,
  input  logic [127:0]             wdata
);
  (* ram_style = RAM_STYLE *) logic [127:0] mem [0:WORDS-1];

  always_ff @(posedge clk) begin
    for (int b = 0; b < 16; b++)
      if (we_byte[b]) mem[waddr][b*8 +: 8] <= wdata[b*8 +: 8];
    rdata <= mem[raddr];
  end
endmodule
`endif
