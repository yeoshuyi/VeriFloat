# Security policy

## Reporting a vulnerability

Please report security problems privately, through GitHub's
[private vulnerability reporting](https://github.com/yeoshuyi/VeriFloat/security/advisories/new),
not in a public issue. Include the VeriFloat version, the platform, and the
smallest input that shows the problem.

You can expect an acknowledgement within a week. Fixes are released as a new
version, with the report credited unless you ask otherwise.

## Scope

VeriFloat runs inside your own Python process or simulator, with your
privileges. What counts as a vulnerability:

- input to a public function (values, shapes, widths, format names, raw
  codes, files read by `verifloat.vectors`) that crashes the interpreter,
  reads or writes memory it should not, or takes memory or time without a
  bound the documentation does not state;
- a C library (`libverifloat`) entry point that does the same with arguments
  that keep to its documented contract;
- anything in the build, the tests or the CI that runs or loads code that was
  not pinned.

Not a vulnerability: unpickling untrusted data (pickle can run arbitrary code
by design), calling private `_core` functions in ways the package never does,
and C callers that break the documented contract (for example, a `_w` buffer
shorter than the format's word count).

## Limits

The documented input limits are listed under
[Limitations](docs/README.md#limitations): at most 64 array axes and 2⁴⁸
elements, integer and fixed-point widths of at most 2²⁴ bits, and FP format
limits on exponent, bias, mantissa and stochastic-rounding widths.
