#!/usr/bin/env python3
"""
Integration test: build LQ preoptimization once, then call the receding-horizon
navigation multiple times from different initial states.

Requirements:
    pip install julia
    python3 -c "import julia; julia.install()"

Run:
    python3 test/test_hardware_integration.py
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

    # Activate and instantiate the Julia project
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

    # Build preoptimization once
    pre = Main.HardwareFunctions.build_lq_preoptimization(3, 0.5, silence_logs=True)

    # Multiple initial states (3 players, 2D each)
    x0_list = [
        [[0.0, 2.0], [2.0, 4.0], [6.0, 8.0]],
        [[0.2, 1.8], [1.9, 3.9], [5.8, 7.7]],
        [[-0.5, 0.5], [1.0, 2.0], [3.0, 4.0]],
    ]

    for idx, x0 in enumerate(x0_list):
        print(f"\n=== RH call {idx} ===")
        try:
            result = Main.HardwareFunctions.hardware_nplayer_hierarchy_navigation(pre, x0, silence_logs=True)
        except Exception as e:
            print("Julia call failed. If error mentions `steps`, the function may need a `steps` argument with a default (e.g., pre.T).\n", e)
            raise

        # PyCall can return a Julia NamedTuple wrapped as a PyCall object which
        # is not subscriptable from Python. Try both styles to extract fields.
        def _get_field(obj, name):
            try:
                return obj[name]
            except Exception:
                try:
                    return getattr(obj, name)
                except Exception:
                    raise RuntimeError(f"Unable to access field '{name}' on Julia result: {type(obj)}")

        states = _get_field(result, "states")
        controls = _get_field(result, "controls")

        # Normalize result contents: PyCall may wrap Julia arrays as numpy arrays
        # or other wrappers; convert recursively to plain Python lists/floats.
        def _normalize(x):
            # If object has a tolist() (numpy arrays), use it
            try:
                tl = x.tolist()
            except Exception:
                tl = None

            if tl is not None:
                return _normalize(tl)

            if isinstance(x, (list, tuple)):
                return [_normalize(xx) for xx in x]

            # Try to cast to float for numeric scalars
            try:
                return float(x)
            except Exception:
                return x

        states = _normalize(states)
        controls = _normalize(controls)

        # Debug: print types and sample contents to diagnose non-finite values
        print("DEBUG: result types ->", type(states), type(controls))
        try:
            print("DEBUG: states repr:", repr(states))
        except Exception:
            print("DEBUG: could not repr states; type:", type(states))
        try:
            print("DEBUG: controls repr:", repr(controls))
        except Exception:
            print("DEBUG: could not repr controls; type:", type(controls))

        # Shape checks
        assert isinstance(states, list) and isinstance(controls, list), "Unexpected result format"
        assert len(states) == 3 and len(controls) == 3, "Expected 3 players"
        for i in range(3):
            assert len(states[i]) >= 2, f"Player {i+1} has too few states"
            assert len(controls[i]) >= 1, f"Player {i+1} has too few controls"

        # Finite checks
        assert finite_nested(states), "Non-finite values in states"
        assert finite_nested(controls), "Non-finite values in controls"

        print(f"states len: {[len(s) for s in states]}")
        print(f"controls len: {[len(u) for u in controls]}")

    print("\nAll integration calls completed successfully.")


if __name__ == "__main__":
    main()
