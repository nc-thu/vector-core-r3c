// glbl.v — Xilinx 全局置位/复位 stub（iverilog 编译 unisim 原语需要）
// GSR 恒 0：本 TB 的复位全部走各模块自己的 rst_n，不依赖全局复位脉冲。
`timescale 1ns/1ps
module glbl;
  wire GSR = 1'b0;
endmodule
