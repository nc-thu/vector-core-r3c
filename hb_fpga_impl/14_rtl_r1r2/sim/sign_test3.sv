module sign_test3;
  // 模拟 ae_actv 的 g_bias mb/prod 逻辑
  logic signed [15:0] rq_m_r;
  logic signed [15:0] m2_r;
  logic        elt_ph2, p1_sq;
  logic signed [7:0] xb;
  logic [7:0] rd_byte;
  
  // 原始 mb（三元有拼接分支 = 无符号）
  wire signed [16:0] mb_orig = elt_ph2 ? m2_r :
                               p1_sq   ? {{9{xb[7]}}, xb} : rq_m_r;
  
  // 修复版 mb（拼接也显式 $signed）
  wire signed [16:0] mb_fix  = elt_ph2 ? m2_r :
                               p1_sq   ? $signed({{9{xb[7]}}, xb}) : rq_m_r;
  
  logic signed [23:0] prod_orig, prod_fix;
  assign prod_orig = $signed(rd_byte) * mb_orig;
  assign prod_fix  = $signed(rd_byte) * mb_fix;

  initial begin
    rq_m_r = -384;  // 16'sd-384
    m2_r = 0;
    elt_ph2 = 0;
    p1_sq = 0;
    xb = 8'sd113;
    rd_byte = 8'd113;
    
    #1;
    $display("rq_m_r = %0d", rq_m_r);
    $display("mb_orig = %0d (unsigned: %0d)", $signed(mb_orig), mb_orig);
    $display("mb_fix  = %0d (unsigned: %0d)", $signed(mb_fix), mb_fix);
    $display("prod_orig = %0d (unsigned: %0d)", $signed(prod_orig), prod_orig);
    $display("prod_fix  = %0d (unsigned: %0d)", $signed(prod_fix), prod_fix);
    
    // 验证：113 * (-384) = -43392
    $display("Expected: %0d", 113 * (-384));
    $finish;
  end
endmodule
