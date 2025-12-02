#!/usr/bin/env python3
"""
Python integration test for `hardware_nplayer_hierarchy_navigation`.

This follows the style of `test_python_integration.py` but exercises the
single-step hardware helper in `examples/hardware_functions.jl`.

It:
- Activates the project
- Includes the required example files into `Main`
- Builds the LQ preoptimization once
- Runs several single-step calls to `hardware_nplayer_hierarchy_navigation`
  and forwards `z_sol` as `z0_guess` to warm-start the next call.

Run:
    python3 test/test_hardware_nplayer_navigation.py

Requirements:
    pip install julia
    python3 -c "import julia; julia.install()"
"""

from pathlib import Path
from julia import Main
import math


def finite_nested(x):
    if isinstance(x, (list, tuple)):
        return all(finite_nested(xx) for xx in x)
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def main():
    project_root = str(Path(__file__).resolve().parents[1])

    # Activate project and include necessary example files
    Main.eval(
        f"""
        import Pkg
        Pkg.activate(raw"{project_root}")
        try
            Pkg.instantiate()
        catch e
            @warn "Pkg.instantiate() failed" exception=e
        end

        using Logging
        global_logger(NullLogger())

    Base.include(Main, joinpath(raw"{project_root}", "examples", "automatic_solver.jl"))
    Base.include(Main, joinpath(raw"{project_root}", "examples", "test_automatic_solver.jl"))
    Base.include(Main, joinpath(raw"{project_root}", "examples", "hardware_functions.jl"))
        """
    )

    # Build preoptimization once (silence logs by default)
    pre = Main.HardwareFunctions.build_lq_preoptimization(3, 0.5, silence_logs=True)

    # initial state for 3 players (2D each)
    x0 = [[0.0, 2.0], [2.0, 4.0], [6.0, 8.0]]

    # number of single-step calls to perform
    steps = 5

    x_current = x0
    z_guess = None

    for step in range(steps):
        print(f"\n--- Step {step} ---")
        print("x_current:", x_current)

        try:
            if z_guess is None:
                result = Main.HardwareFunctions.hardware_nplayer_hierarchy_navigation(pre, x_current, silence_logs=True)
            else:
                result = Main.HardwareFunctions.hardware_nplayer_hierarchy_navigation(pre, x_current, z_guess, silence_logs=True)
        except Exception as e:
            print("Julia call failed:")
            raise

        # helper to access PyCall-wrapped NamedTuple fields
        def _get_field(obj, name):
            try:
                return obj[name]
            except Exception:
                return getattr(obj, name)

        x_next = _get_field(result, "x_next")
        u_curr = _get_field(result, "u_curr")
        try:
            z_guess = _get_field(result, "z_sol")
        except Exception:
            z_guess = None

        # normalize
        def _normalize(x):
            try:
                tl = x.tolist()
            except Exception:
                tl = None
            if tl is not None:
                return _normalize(tl)
            if isinstance(x, (list, tuple)):
                return [_normalize(xx) for xx in x]
            try:
                return float(x)
            except Exception:
                return x

        x_next = _normalize(x_next)
        u_curr = _normalize(u_curr)

        print("u_curr:", u_curr)
        print("x_next:", x_next)

        # basic checks
        assert isinstance(x_next, list) and isinstance(u_curr, list)
        assert finite_nested(x_next)
        assert finite_nested(u_curr)

        x_current = x_next

    print("\nAll hardware_nplayer_hierarchy_navigation calls completed successfully.")


if __name__ == "__main__":
    main()