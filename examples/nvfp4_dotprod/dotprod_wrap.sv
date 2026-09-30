// Flattens dotprod_top's unpacked-array ports into buses for cocotb.
// Lane i of a_flat/b_flat is a[i]/b[i] (16 bits each, lane 0 in the LSBs).
module dotprod_wrap
  import dotprod_pkg::*;
(
  input  logic [1:0]                mode,
  input  logic [16*N_LANES-1:0]     a_flat,
  input  logic [16*N_LANES-1:0]     b_flat,
  output logic [FP32_W-1:0]         result,
  output logic [3:0]                status     // {sat, invalid, is_nan, is_inf}
);
  logic [BF16_W-1:0] a [N_LANES], b [N_LANES];
  dotprod_status_t   st;
  logic              sat;

  always_comb begin
    for (int i = 0; i < N_LANES; i++) begin
      a[i] = a_flat[16*i +: 16];
      b[i] = b_flat[16*i +: 16];
    end
  end

  dotprod_top u_top (.mode(fmt_e'(mode)), .a(a), .b(b), .result(result),
                     .status(st), .sat(sat));
  assign status = st;
endmodule
