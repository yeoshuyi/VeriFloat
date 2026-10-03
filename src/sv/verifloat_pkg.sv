// VeriFloat for SystemVerilog: the bit-accurate floating-point model through
// DPI-C. Link the simulation with libverifloat (see `python -m verifloat.dpi`).
//
//   import verifloat_pkg::*;
//   chandle fp32 = vf_format("e8m23");            // once; see verifloat.h for names
//   int flags;
//   expected = vf_add(fp32, a, b, flags);         // raw code and IEEE flags
//
// Codes are raw encodings [sign | exponent | mantissa] in the low bits of a
// 64-bit value; `flags` uses the bit order of RISC-V fflags. When the model
// has no answer, the result is 0, flags is VF_ERROR and vf_last_error() says
// why. Formats wider than 64 bits: vf_op128 and its relatives below.
package verifloat_pkg;

  typedef longint unsigned vf_code_t;

  // Status flags
  localparam int VF_INEXACT   = 1;
  localparam int VF_UNDERFLOW = 2;
  localparam int VF_OVERFLOW  = 4;
  localparam int VF_DIVZERO   = 8;
  localparam int VF_INVALID   = 16;
  localparam int VF_ERROR     = 256;

  // Rounding modes (vf_to_int, vf_round_to_integral)
  localparam int VF_ROUND_FORMAT = -1;
  localparam int VF_RNE = 0;
  localparam int VF_RNA = 1;
  localparam int VF_RTZ = 2;
  localparam int VF_RUP = 3;
  localparam int VF_RDN = 4;
  localparam int VF_SR  = 5;

  // vf_compare results
  localparam int VF_LT = 0;
  localparam int VF_EQ = 1;
  localparam int VF_GT = 2;
  localparam int VF_UNORDERED = 3;

  // Operation numbers (vf_op, vf_op128)
  localparam int VF_OP_ADD = 0;
  localparam int VF_OP_SUB = 1;
  localparam int VF_OP_MUL = 2;
  localparam int VF_OP_DIV = 3;
  localparam int VF_OP_REM = 4;
  localparam int VF_OP_FMOD = 5;
  localparam int VF_OP_MIN = 6;
  localparam int VF_OP_MAX = 7;
  localparam int VF_OP_MINNUM = 8;
  localparam int VF_OP_MAXNUM = 9;
  localparam int VF_OP_SGNJ = 10;
  localparam int VF_OP_SGNJN = 11;
  localparam int VF_OP_SGNJX = 12;
  localparam int VF_OP_FMA = 13;
  localparam int VF_OP_SQRT = 14;
  localparam int VF_OP_NEG = 15;
  localparam int VF_OP_ABS = 16;
  localparam int VF_OP_NEXT_UP = 17;
  localparam int VF_OP_NEXT_DOWN = 18;
  localparam int VF_OP_LOGB = 19;

  // Formats
  import "DPI-C" function chandle vf_format(input string name);
  import "DPI-C" function int vf_format_size(input chandle f);
  import "DPI-C" function string vf_format_name(input chandle f);
  import "DPI-C" function string vf_last_error();
  import "DPI-C" function string vf_version();

  // Arithmetic
  import "DPI-C" function vf_code_t vf_add(input chandle f, input vf_code_t a, input vf_code_t b, output int flags);
  import "DPI-C" function vf_code_t vf_sub(input chandle f, input vf_code_t a, input vf_code_t b, output int flags);
  import "DPI-C" function vf_code_t vf_mul(input chandle f, input vf_code_t a, input vf_code_t b, output int flags);
  import "DPI-C" function vf_code_t vf_div(input chandle f, input vf_code_t a, input vf_code_t b, output int flags);
  import "DPI-C" function vf_code_t vf_fma(input chandle f, input vf_code_t a, input vf_code_t b, input vf_code_t c,
                                           output int flags);
  import "DPI-C" function vf_code_t vf_sqrt(input chandle f, input vf_code_t a, output int flags);
  import "DPI-C" function vf_code_t vf_rem(input chandle f, input vf_code_t a, input vf_code_t b, output int flags);
  import "DPI-C" function vf_code_t vf_fmod(input chandle f, input vf_code_t a, input vf_code_t b, output int flags);
  import "DPI-C" function vf_code_t vf_min(input chandle f, input vf_code_t a, input vf_code_t b, output int flags);
  import "DPI-C" function vf_code_t vf_max(input chandle f, input vf_code_t a, input vf_code_t b, output int flags);
  import "DPI-C" function vf_code_t vf_minnum(input chandle f, input vf_code_t a, input vf_code_t b,
                                              output int flags);
  import "DPI-C" function vf_code_t vf_maxnum(input chandle f, input vf_code_t a, input vf_code_t b,
                                              output int flags);
  import "DPI-C" function vf_code_t vf_neg(input chandle f, input vf_code_t a, output int flags);
  import "DPI-C" function vf_code_t vf_abs(input chandle f, input vf_code_t a, output int flags);
  import "DPI-C" function vf_code_t vf_sgnj(input chandle f, input vf_code_t a, input vf_code_t b, output int flags);
  import "DPI-C" function vf_code_t vf_sgnjn(input chandle f, input vf_code_t a, input vf_code_t b,
                                             output int flags);
  import "DPI-C" function vf_code_t vf_sgnjx(input chandle f, input vf_code_t a, input vf_code_t b,
                                             output int flags);
  import "DPI-C" function vf_code_t vf_next_up(input chandle f, input vf_code_t a, output int flags);
  import "DPI-C" function vf_code_t vf_next_down(input chandle f, input vf_code_t a, output int flags);
  import "DPI-C" function vf_code_t vf_scaleb(input chandle f, input vf_code_t a, input longint n, output int flags);
  import "DPI-C" function vf_code_t vf_logb(input chandle f, input vf_code_t a, output int flags);
  import "DPI-C" function int vf_fclass(input chandle f, input vf_code_t a);
  import "DPI-C" function vf_code_t vf_op(input chandle f, input int op, input vf_code_t a, input vf_code_t b,
                                          input vf_code_t c, output int flags);

  // Comparisons (signaling: a quiet NaN raises VF_INVALID too)
  import "DPI-C" function int vf_compare(input chandle f, input vf_code_t a, input vf_code_t b, input int signaling,
                                         output int flags);
  import "DPI-C" function int vf_eq(input chandle f, input vf_code_t a, input vf_code_t b, input int signaling,
                                    output int flags);
  import "DPI-C" function int vf_lt(input chandle f, input vf_code_t a, input vf_code_t b, input int signaling,
                                    output int flags);
  import "DPI-C" function int vf_le(input chandle f, input vf_code_t a, input vf_code_t b, input int signaling,
                                    output int flags);

  // Conversions
  import "DPI-C" function vf_code_t vf_convert(input chandle to, input chandle from, input vf_code_t a,
                                               output int flags);
  import "DPI-C" function vf_code_t vf_round_to_integral(input chandle f, input vf_code_t a, input int rounding,
                                                         input int exact, output int flags);
  import "DPI-C" function vf_code_t vf_to_int(input chandle f, input vf_code_t a, input int bits, input int is_signed,
                                              input int rounding, input int exact, output int flags);
  import "DPI-C" function vf_code_t vf_from_int(input chandle f, input longint value, output int flags);
  import "DPI-C" function vf_code_t vf_from_uint(input chandle f, input longint unsigned value, output int flags);
  import "DPI-C" function vf_code_t vf_from_double(input chandle f, input real value, output int flags);
  import "DPI-C" function real vf_to_double(input chandle f, input vf_code_t a);

  // Stochastic rounding: the random bits the next roundings use
  import "DPI-C" function void vf_set_sr(input longint unsigned bits);

  // Formats of up to 128 bits (binary128 is "e15m112"): codes in the low bits
  // of a bit [127:0], the result zero above the format's width. For wider
  // formats, import vf_op_w and its relatives (verifloat.h) with your own
  // vector width.
  import "DPI-C" function void vf_op128(input chandle f, input int op, input bit [127:0] a, input bit [127:0] b,
                                        input bit [127:0] c, output bit [127:0] result, output int flags);
  import "DPI-C" function void vf_scaleb128(input chandle f, input bit [127:0] a, input longint n,
                                            output bit [127:0] result, output int flags);
  import "DPI-C" function int vf_fclass128(input chandle f, input bit [127:0] a);
  import "DPI-C" function int vf_compare128(input chandle f, input bit [127:0] a, input bit [127:0] b,
                                            input int signaling, output int flags);
  import "DPI-C" function void vf_convert128(input chandle to, input chandle from, input bit [127:0] a,
                                             output bit [127:0] result, output int flags);
  import "DPI-C" function void vf_round_to_integral128(input chandle f, input bit [127:0] a, input int rounding,
                                                       input int exact, output bit [127:0] result,
                                                       output int flags);

  // Text for a flags value, e.g. "INEXACT|OVERFLOW".
  function automatic string vf_flags_str(input int flags);
    string s = "";
    if (flags & VF_ERROR) return "ERROR";
    if (flags & VF_INVALID) s = {s, "|INVALID"};
    if (flags & VF_DIVZERO) s = {s, "|DIVZERO"};
    if (flags & VF_OVERFLOW) s = {s, "|OVERFLOW"};
    if (flags & VF_UNDERFLOW) s = {s, "|UNDERFLOW"};
    if (flags & VF_INEXACT) s = {s, "|INEXACT"};
    if (s == "") return "none";
    return s.substr(1, s.len() - 1);
  endfunction

endpackage
