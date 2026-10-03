"""The public API of the C++-backed package against the frozen pure-Python
0.1 (archive/python): the same names, the same call signatures, the same
class layout. tests/test_parity.py compares what the calls return; this file
compares what can be called.

The only additions are ``FPArray`` and ``FPFormat.array``. The one accepted
difference: ``inspect.signature`` of a format *instance* (``FP16(...)``) shows
``(*args, **kwargs)``, because calling a format is a native slot; the call
itself takes the same arguments (test_parity.py::test_keyword_arguments).
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import os

import pytest

if os.environ.get("VERIFLOAT_IMPL", "cpp") != "cpp":
    pytest.skip("compares the C++ package against verifloat_py", allow_module_level=True)

import verifloat as V
import verifloat.bus  # noqa: F401

P = pytest.importorskip("verifloat_py")
import verifloat_py.bus  # noqa: E402,F401

ADDED = {"FPArray"}
ADDED_ATTRS = {"FPFormat": {"array"},
               # format fields: reached through __getattr__ in 0.1, real attributes now
               "FP": {"bias", "exp_bits", "ftz", "has_zero", "inf_nan", "mantissa_bits", "nan_mode", "rounding",
                      "saturate", "signed", "sr_bits", "tininess", "wrap",
                      # IEEE 754 operations added after 0.1 (tests/test_ieee_ops.py)
                      "remainder", "fmod", "next_up", "next_down", "scaleb", "logb", "fclass", "is_normal", "copysign",
                      "fsgnj", "fsgnjn", "fsgnjx"}}
MODULES = ["fp", "uint", "sint", "fixed", "accum", "blockscale", "bus", "_warnings"]
# Standard-library names 0.1 happened to import into its modules.
INCIDENTAL = {"math", "total_ordering", "annotations"}


def public(obj) -> set[str]:
    return {n for n in dir(obj) if not n.startswith("_")}


def params(sig: inspect.Signature, drop_self: bool):
    """(name, kind, default) of each parameter. A native method reports
    ``self`` as positional-only; that parameter is never passed by name."""
    ps = list(sig.parameters.values())
    if drop_self and ps and ps[0].name in ("self", "cls"):
        ps = ps[1:]
    return [(p.name, p.kind.name, None if p.default is p.empty else repr(p.default)) for p in ps]


def signature(obj):
    try:
        return inspect.signature(obj)
    except (TypeError, ValueError):
        return None


def test_exported_names():
    assert set(V.__all__) == set(P.__all__) | ADDED
    assert V.__all__[:0] == [] and len(set(V.__all__)) == len(V.__all__)
    for name in P.__all__:
        assert hasattr(V, name), name
    assert V.__version__ == P.__version__


@pytest.mark.parametrize("mod", MODULES)
def test_module_names(mod):
    """Everything importable from a 0.1 submodule is still importable."""
    a, b = getattr(V, mod), getattr(P, mod)
    missing = public(b) - public(a) - INCIDENTAL
    assert not missing, f"verifloat.{mod} lacks {sorted(missing)}"
    for name in public(b) - INCIDENTAL:
        x, y = getattr(a, name), getattr(b, name)
        assert inspect.isclass(x) == inspect.isclass(y) and callable(x) == callable(y), name


@pytest.mark.parametrize("name", [n for n in P.__all__ if inspect.isclass(getattr(P, n))])
def test_class_surface(name):
    a, b = getattr(V, name), getattr(P, name)
    assert a.__name__ == b.__name__ and a.__qualname__ == b.__qualname__
    assert a.__module__ == b.__module__.replace("verifloat_py", "verifloat")
    assert public(a) == public(b) | ADDED_ATTRS.get(name, set())
    # the special methods 0.1 defined (operators, conversions, hashing...)
    dunders = {n for n in vars(b) if n.startswith("__") and n.endswith("__")} - {
        "__slots__", "__dict__", "__weakref__", "__module__", "__doc__", "__getattr__", "__init__",
        "__annotations__", "__dataclass_fields__", "__dataclass_params__", "__match_args__",
        "__firstlineno__", "__static_attributes__", "__abstractmethods__", "__orig_bases__",
        "__parameters__"}
    for d in dunders:
        if d.startswith("__r") and d[3:] != "epr__" and not d.startswith("__reduce"):
            continue    # reflected operators: a native type implements them in the forward slot
        assert hasattr(a, d) and (getattr(a, d) is None) == (getattr(b, d) is None), f"{name}.{d}"
    for attr in sorted(public(b)):
        x, y = inspect.getattr_static(a, attr), inspect.getattr_static(b, attr)
        data_y = isinstance(y, property) or not callable(getattr(b, attr))
        data_x = inspect.isdatadescriptor(x) or not callable(getattr(a, attr))
        assert data_x == data_y, f"{name}.{attr}: attribute vs method"
        if data_y:
            continue
        if not inspect.isroutine(getattr(b, attr)):
            continue            # a callable value (a format): compared by the parity tests
        sa, sb = signature(getattr(a, attr)), signature(getattr(b, attr))
        if sb is None:          # inherited from a builtin (int.to_bytes, Warning.with_traceback)
            continue
        assert sa is not None, f"{name}.{attr} has no signature"
        assert params(sa, True) == params(sb, True), f"{name}.{attr}{sa} != {sb}"
    if dataclasses.is_dataclass(b):
        fa, fb = dataclasses.fields(a), dataclasses.fields(b)
        assert [(f.name, f.kw_only, repr(f.default)) for f in fa] == \
               [(f.name, f.kw_only, repr(f.default)) for f in fb]
        assert dataclasses.is_dataclass(a) and a.__dataclass_params__.frozen == b.__dataclass_params__.frozen
    if issubclass(b, enum.Enum):
        assert [(m.name, m.value) for m in a] == [(m.name, m.value) for m in b]
        assert [c.__name__ for c in a.__mro__] == [c.__name__ for c in b.__mro__]
    if issubclass(b, Warning):
        assert [c.__name__ for c in a.__mro__] == [c.__name__ for c in b.__mro__]
    # constructor signature
    sa, sb = signature(a), signature(b)
    if sb is not None and name != "FIXED":      # FIXED values are built by a FixedFormat
        assert sa is not None, f"{name}() has no signature"
        assert params(sa, False) == params(sb, False), f"{name}{sa} != {sb}"


@pytest.mark.parametrize("name", [n for n in P.__all__ if not inspect.isclass(getattr(P, n))])
def test_functions_and_constants(name):
    a, b = getattr(V, name), getattr(P, name)
    assert type(a).__name__ == type(b).__name__
    if inspect.isfunction(b):
        assert params(signature(a), False) == params(signature(b), False)
        assert a.__name__ == b.__name__
    else:                                       # format presets
        assert str(a) == str(b) and repr(a) == repr(b)
        assert public(a) == public(b) | ADDED_ATTRS.get(type(b).__name__, set())


def test_bus_functions():
    for name in public(P.bus):
        a, b = getattr(V.bus, name), getattr(P.bus, name)
        if inspect.isfunction(b):
            assert params(signature(a), False) == params(signature(b), False), name


def test_instances_look_the_same():
    """isinstance relations, attribute sets and type names of live values."""
    def values(m):
        return [m.UINT(5, 3), m.INT(-2, 4), m.FP16(0.1), m.E4M3(1.5), m.FixedFormat(4, 4)(1.25),
                m.FP16, m.FixedFormat(4, 4), m.NVFP4, m.Accumulator(m.FP32), m.IntFormat(8),
                m.Pow2Format(8, 127), m.BlockTensor.quantize([1.0, 2.0, 3.0, 4.0]), m.Rounding.RNE,
                m.FPFlags.INEXACT | m.FPFlags.OVERFLOW, m.NaNMode.ARM]
    for x, y in zip(values(V), values(P)):
        assert type(x).__name__ == type(y).__name__
        assert public(x) == public(y) | ADDED_ATTRS.get(type(y).__name__, set()), type(y).__name__
        assert isinstance(x, getattr(V, type(y).__name__))
    assert isinstance(V.INT(1, 2), V.UINT) == isinstance(P.INT(1, 2), P.UINT)
    assert issubclass(V.INT, V.UINT) and issubclass(V.FPUnderflowWarning, V.VeriFloatWarning)
    assert isinstance(V.FP16(1), V.FP) and isinstance(V.FP16, V.FPFormat)
    assert V.FP16(1).format is V.FP16 or V.FP16(1).format == V.FP16


def test_docstrings_kept():
    """Every documented public method of 0.1 is still documented."""
    for name in P.__all__:
        b = getattr(P, name)
        if not inspect.isclass(b):
            continue
        a = getattr(V, name)
        if b.__doc__:
            assert a.__doc__, name
        for attr in public(b):
            if getattr(getattr(b, attr), "__doc__", None) and callable(getattr(b, attr)) \
                    and attr in vars(b):
                assert getattr(a, attr).__doc__, f"{name}.{attr}"
