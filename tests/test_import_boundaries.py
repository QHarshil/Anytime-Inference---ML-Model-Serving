"""Guards on the serving control plane's dependency surface.

The control plane is meant to run wherever the C++ runtime runs, which is not
necessarily a machine with the research stack installed. Both torch and pandas
are heavy optional dependencies, and both have previously leaked into the
serving import path: torch through a module-scope import in ``models.cascade``,
pandas through ``utils/__init__`` eagerly re-exporting ``utils.io``.

These tests import the serving modules in a subprocess and assert the heavy
dependencies were never loaded. A subprocess is required because the modules are
already imported in the parent process by the time this file runs.
"""

import subprocess
import sys
import textwrap

# Modules that must import without pulling in torch or pandas.
SERVING_MODULES = (
    "anytime_serving.serving",
    "anytime_serving.serving.admission",
    "anytime_serving.serving.decoder",
    "anytime_serving.serving.kv_admission",
    "anytime_serving.serving.load_monitor",
    "anytime_serving.serving.onnx_runtime",
    "anytime_serving.serving.selector",
    "anytime_serving.serving.server",
    "anytime_serving.utils.logger",
    "anytime_serving.models.cascade",
)

FORBIDDEN = ("torch", "pandas")


def _import_and_report(modules: tuple[str, ...]) -> set[str]:
    """Import *modules* in a clean interpreter, returning forbidden loads."""
    script = textwrap.dedent(f"""
        import sys
        for name in {list(modules)!r}:
            __import__(name)
        leaked = [dep for dep in {list(FORBIDDEN)!r} if dep in sys.modules]
        print(",".join(leaked))
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return {name for name in completed.stdout.strip().split(",") if name}


def test_serving_modules_do_not_import_torch_or_pandas():
    leaked = _import_and_report(SERVING_MODULES)
    assert not leaked, (
        f"serving import path pulled in {sorted(leaked)}. Move the import inside "
        f"the functions that need it, or behind typing.TYPE_CHECKING."
    )


def test_utils_package_import_stays_light():
    """Importing the utils package must not cost pandas."""
    leaked = _import_and_report(("anytime_serving.utils",))
    assert not leaked, f"anytime_serving.utils pulled in {sorted(leaked)}"


def test_lazy_utils_exports_stay_advertised():
    """The pandas-backed names remain visible in the package namespace.

    Checked without resolving them, so the base install exercises this too. What
    the ``__getattr__`` hook has to preserve is that the namespace looks
    unchanged, and ``dir()`` plus ``__all__`` show that without importing pandas.
    """
    from anytime_serving import utils

    for name in ("save_csv", "compute_hit_rate"):
        assert name in dir(utils)
        assert name in utils.__all__


def test_lazy_utils_exports_resolve_to_callables():
    """Resolving a lazy name yields the helper itself, not just the name.

    Every lazy export is pandas-backed, so this cannot run in the base install
    the ``test-minimal`` CI job builds: that job asserts pandas is absent. The
    advertised-names test above is what covers the hook there.
    """
    import pytest

    pytest.importorskip("pandas")

    from anytime_serving import utils

    assert callable(utils.save_csv)
    assert callable(utils.compute_hit_rate)


def test_unknown_utils_attribute_raises_attribute_error():
    import pytest

    from anytime_serving import utils

    with pytest.raises(AttributeError, match="does_not_exist"):
        _ = utils.does_not_exist
