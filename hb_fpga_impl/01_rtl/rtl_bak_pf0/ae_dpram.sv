// ae_dpram.sv — 真双端口 RAM（BRAM/URAM 原生 TDP）
// A 口：GEMM A-流读；B 口：B-流读 / 写回 / softmax / DMA（调度器串行化，无冲突）
`ifndef AE_DPRAM_SV
`define AE_DPRAM_SV
module ae_dpram #(
  parameter int WIDTH = 8,
  parameter int WORDS = 4096,
  parameter int AW    = $clog2(WORDS),
  parameter       RAM_STYLE = "block"
)(
  input  logic               clk,
  input  logic               a_we,
  input  logic [AW-1:0]      a_addr,
  input  logic [WIDTH-1:0]   a_wdata,
  output logic [WIDTH-1:0]   a_rdata,
  input  logic               b_we,
  input  logic [AW-1:0]      b_addr,
  input  logic [WIDTH-1:0]   b_wdata,
  output logic [WIDTH-1:0]   b_rdata
);
  (* ram_style = RAM_STYLE *) logic [WIDTH-1:0] mem [0:WORDS-1];
  initial begin
    for (int i = 0; i < WORDS; i++) mem[i] = '0;
  end

  always_ff @(posedge clk) begin
    if (a_we) mem[a_addr] <= a_wdata;
    a_rdata <= mem[a_addr];
  end
  always_ff @(posedge clk) begin
    if (b_we) mem[b_addr] <= b_wdata;
    b_rdata <= mem[b_addr];
  end
endmodule
`endif
