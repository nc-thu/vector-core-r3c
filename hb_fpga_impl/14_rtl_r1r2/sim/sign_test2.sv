module sign_test2;
  logic [15:0] unsigned_in;
  logic signed [15:0] signed_reg;
  
  initial begin
    // 模拟 ae_actv 的 A_IDLE 赋值
    unsigned_in = 16'hFE80;
    signed_reg = unsigned_in;  // 直接赋值
    
    #1;
    $display("Test 1: direct assign");
    $display("  unsigned_in = %0d (0x%04X)", unsigned_in, unsigned_in);
    $display("  signed_reg = %0d", signed_reg);
    
    // 测试：在 always_ff 中赋值
    signed_reg = 0;
    #1;
  end
  
  // 用 always_ff 赋值（模拟 RTL 行为）
  logic clk = 0;
  always #5 clk = ~clk;
  logic trigger = 0;
  
  always_ff @(posedge clk) begin
    if (trigger) begin
      signed_reg <= unsigned_in;
      $display("  [always_ff] signed_reg assigned, now = %0d", signed_reg);
    end
  end
  
  initial begin
    trigger = 0;
    #2;
    $display("Test 2: always_ff assign");
    unsigned_in = 16'hFE80;
    #2;
    trigger = 1;
    #10;
    $display("  After clock: signed_reg = %0d", signed_reg);
    
    // 测试 3: 在 case 语句中赋值
    $display("Test 3: signed vs unsigned context");
    $display("  $signed(unsigned_in) = %0d", $signed(unsigned_in));
    $display("  signed_reg + 0 = %0d", signed_reg + 0);
    
    $finish;
  end
endmodule
