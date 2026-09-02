module sign_test5;
  logic [127:0] rd_r;
  logic signed [15:0] rq_m_r;
  logic signed [15:0] b_r;
  logic        elt_ph2, p1_sq, md_elt;
  logic signed [15:0] m2_r;
  logic signed [24:0] ea_q;
  logic [7:0]  rq_s_r;
  logic        md_norm;
  logic [4:0]  st_dummy;
  
  logic signed [23:0] pmul [0:15];
  for (genvar g = 0; g < 16; g++) begin : g_bias
    (* use_dsp = "no" *) logic signed [23:0] prod;
    logic signed [24:0] accb, p_sh;
    logic signed [24:0] accb_q;
    wire signed [7:0]  xb = rd_r[8*g +: 8];
    wire signed [16:0] mb = elt_ph2 ? m2_r :
                            p1_sq   ? $signed({{9{xb[7]}}, xb}) : rq_m_r;
    assign prod  = $signed(rd_r[8*g +: 8]) * mb;
    assign pmul[g] = prod;
    assign accb  = prod + (md_elt ? ea_q : $signed({{9{b_r[15]}}, b_r}));
    always_ff @(posedge rq_s_r[0]) accb_q <= accb;
    assign p_sh  = accb_q >>> rq_s_r;
  end

  initial begin
    rd_r = {120'b0, 8'd113};  // lane 0 = 113
    rq_m_r = -384;
    b_r = 1764;
    elt_ph2 = 0;
    p1_sq = 0;
    md_elt = 0;
    md_norm = 0;
    ea_q = 0;
    rq_s_r = 8'd4;
    
    #1;
    $display("g_bias[0].xb = %0d", g_bias[0].xb);
    $display("g_bias[0].mb = %0d", g_bias[0].mb);
    $display("g_bias[0].prod = %0d", g_bias[0].prod);
    $display("g_bias[0].accb = %0d", g_bias[0].accb);
    $display("Expected prod = %0d", 113 * (-384));
    $finish;
  end
endmodule
