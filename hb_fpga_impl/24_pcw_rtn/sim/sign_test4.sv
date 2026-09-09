module sign_test4;
  logic signed [15:0] rq_m_r;
  logic signed [15:0] m2_r;
  logic        elt_ph2, p1_sq;
  logic signed [7:0] xb;
  logic [7:0] rd_byte;
  
  // 测试: $signed(concat) 在三元里
  wire signed [16:0] mb1 = elt_ph2 ? m2_r :
                           p1_sq   ? $signed({{9{xb[7]}}, xb}) : rq_m_r;
  
  // 测试: 先落 wire signed 再用
  wire signed [16:0] xse = $signed({{9{xb[7]}}, xb});
  wire signed [16:0] mb2 = elt_ph2 ? m2_r :
                           p1_sq   ? xse : rq_m_r;
  
  // 测试: 不用三元，直接 if-else in always
  logic signed [16:0] mb3;
  always_comb begin
    if (elt_ph2) mb3 = m2_r;
    else if (p1_sq) mb3 = $signed({{9{xb[7]}}, xb});
    else mb3 = rq_m_r;
  end
  
  logic signed [23:0] prod1, prod2, prod3;
  assign prod1 = $signed(rd_byte) * mb1;
  assign prod2 = $signed(rd_byte) * mb2;
  assign prod3 = $signed(rd_byte) * mb3;

  initial begin
    rq_m_r = -384;
    m2_r = 0;
    elt_ph2 = 0;
    p1_sq = 0;
    xb = 8'sd113;
    rd_byte = 8'd113;
    
    #1;
    $display("mb1=%0d prod1=%0d", mb1, prod1);
    $display("mb2=%0d prod2=%0d", mb2, prod2);
    $display("mb3=%0d prod3=%0d", mb3, prod3);
    $display("Expected: mb=-384 prod=%0d", 113*(-384));
    $finish;
  end
endmodule
