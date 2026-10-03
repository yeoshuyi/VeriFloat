// SystemVerilog testbench: the dawsonjon/fpu single-precision adder against
// VeriFloat, called through DPI-C. No Python runs during the simulation.
//
//   +n=<vectors>  +seed=<seed>  +rounding=<tag>   (e.g. +rounding=rtz to see it fail)
module tb;
  import verifloat_pkg::*;

  logic clk = 0, rst = 1;
  logic [31:0] a, b;
  logic a_stb = 0, b_stb = 0;
  wire a_ack, b_ack, z_stb;
  wire [31:0] z;

  adder dut (.clk(clk), .rst(rst), .input_a(a), .input_a_stb(a_stb), .input_a_ack(a_ack),
             .input_b(b), .input_b_stb(b_stb), .input_b_ack(b_ack),
             .output_z(z), .output_z_stb(z_stb), .output_z_ack(1'b1));

  always #5 clk = ~clk;

  // Constrained random binary32 encodings: specials, subnormals, operands of
  // similar magnitude (cancellation) and uniformly random bits.
  function automatic logic [31:0] stimulus();
    int unsigned r = $urandom_range(99);
    logic sign = $urandom_range(1)[0];
    logic [31:0] special [8] = '{32'h0, 32'h1, 32'h007FFFFF, 32'h00800000, 32'h7F7FFFFF, 32'h7F800000,
                                 32'h7FC00000, 32'h3F800000};
    if (r < 10) return {sign, special[$urandom_range(7)][30:0]};
    if (r < 50) return {sign, 8'($urandom_range(134, 120)), 23'($urandom)};
    if (r < 60) return {sign, 8'($urandom_range(2)), 23'($urandom)};
    return $urandom;
  endfunction

  initial begin   // watchdog: the adder answers within a few hundred cycles
    automatic longint cycles = 0, last = -1;
    forever begin
      @(posedge clk);
      cycles++;
      if (z_stb) last = cycles;
      if (cycles - last > 5000 && last >= 0 || cycles > 5000 && last < 0) $fatal(1, "no result for 5000 cycles");
    end
  end

  localparam int VF_NAN = (1 << 8) | (1 << 9);   // vf_fclass: signaling or quiet NaN

  initial begin
    chandle fp32;
    int n, seed, flags, mismatches;
    string rounding, name;
    longint unsigned want;
    if (!$value$plusargs("n=%d", n)) n = 2000;
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    name = "e8m23";
    if ($value$plusargs("rounding=%s", rounding)) name = {name, ", ", rounding};
    fp32 = vf_format(name);                     // the golden model's format
    if (fp32 == null) $fatal(1, "vf_format: %s", vf_last_error());
    void'($urandom(seed));

    // Inputs change on the falling edge; the adder samples on the rising one.
    repeat (3) @(negedge clk);
    rst = 0;
    for (int i = 0; i < n; i++) begin
      a = stimulus();
      a_stb = 1;
      do @(negedge clk); while (!a_ack);    // ack with stb high: taken at the next rising edge
      @(negedge clk);
      a_stb = 0;
      b = stimulus();
      b_stb = 1;
      do @(negedge clk); while (!b_ack);
      @(negedge clk);
      b_stb = 0;
      while (!z_stb) @(negedge clk);

      want = vf_add(fp32, 64'(a), 64'(b), flags);
      // The adder returns a negative quiet NaN (0xFFC00000) where the model's
      // canonical NaN is 0x7FC00000: any NaN is accepted for a NaN result.
      // It has no flag outputs, so `flags` is not compared.
      if ((vf_fclass(fp32, want) & VF_NAN) != 0 ? (vf_fclass(fp32, 64'(z)) & VF_NAN) == 0 : z != want[31:0]) begin
        mismatches++;
        if (mismatches <= 10)
          $display("MISMATCH %08h + %08h: dut %08h, model %08h [%s]", a, b, z, want[31:0], vf_flags_str(flags));
      end
      @(negedge clk);
    end
    $display("%s: %0d vectors, %0d mismatches (seed %0d, VeriFloat %s)", vf_format_name(fp32), n, mismatches,
             seed, vf_version());
    if (mismatches != 0) $fatal(1, "adder does not match the model");
    $finish;
  end
endmodule
