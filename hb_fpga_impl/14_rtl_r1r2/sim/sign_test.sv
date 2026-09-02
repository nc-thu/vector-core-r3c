module sign_test;
  logic [15:0] unsigned_in;
  logic signed [15:0] signed_reg;
  logic signed [16:0] mb;
  logic signed [23:0] prod;
  logic signed [23:0] prod2;
  logic signed [7:0] x;

  initial begin
    unsigned_in = 16'hFE80;
    signed_reg = unsigned_in;
    $display("unsigned_in = %0d (0x%04X)", unsigned_in, unsigned_in);
    $display("signed_reg = %0d", signed_reg);
    
    mb = signed_reg;
    $display("mb = %0d", mb);
    
    x = 8'sd85;
    prod = $signed(x) * mb;
    $display("prod = %0d", prod);
    
    prod2 = $signed(x) * $signed(unsigned_in);
    $display("prod2 (direct $signed) = %0d", prod2);
    
    $finish;
  end
endmodule
