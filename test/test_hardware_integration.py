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
    # TODO: Generate the x0_list programmatically using the results of each call in the loop.
    x0_list = [
        [[0.0, 2.0], [2.0, 4.0], [6.0, 8.0]],
        [[0.2, 1.8], [1.9, 3.9], [5.8, 7.7]],
        [[-0.5, 0.5], [1.0, 2.0], [3.0, 4.0]],
    ]
    # For each initial state, run a receding-horizon loop in Python calling the
    # single-step hardware function once per timestep and updating x0.
    for idx, x0 in enumerate(x0_list):
        print(f"\n=== RH call {idx} ===")

        # histories per player
        Nplayers = len(x0)
        states_hist = [[] for _ in range(Nplayers)]
        controls_hist = [[] for _ in range(Nplayers)]

        # initial state
        x_current = x0

        # determine number of steps from pre (pre.T) if available, else default to 3
        try:
            steps = int(Main.getproperty(pre, "T"))
        except Exception:
            steps = 3

        # optional warm-start guess for internal solver variables
        z_guess = None

        for step in range(steps):
            try:
                # pass previous z_sol as z0_guess to warm-start the solver
                if z_guess is None:
                    result = Main.HardwareFunctions.hardware_nplayer_hierarchy_navigation(pre, x_current, silence_logs=True)
                else:
                    result = Main.HardwareFunctions.hardware_nplayer_hierarchy_navigation(pre, x_current, z_guess, silence_logs=True)
            except Exception as e:
                print("Julia call failed during step loop:\n", e)
                raise

            # extract fields (result is a PyCall wrapper for a NamedTuple)
            def _get_field(obj, name):
                try:
                    return obj[name]
                except Exception:
                    return getattr(obj, name)

            x_next = _get_field(result, "x_next")
            u_curr = _get_field(result, "u_curr")
            # capture z_sol to use as z0_guess on next iteration
            try:
                # TODO: Adjust z_sol to use x0 as the primal guess.
                z_guess = _get_field(result, "z_sol")
            except Exception:
                z_guess = None

            # normalize to python lists
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

            # append histories
            for i in range(Nplayers):
                states_hist[i].append(x_current[i])
                controls_hist[i].append(u_curr[i])

            # advance state
            x_current = x_next

        # Final checks
        assert all(len(states_hist[i]) >= 1 for i in range(Nplayers)), "Too few states"
        assert all(len(controls_hist[i]) >= 1 for i in range(Nplayers)), "Too few controls"
        assert finite_nested(states_hist), "Non-finite values in states"
        assert finite_nested(controls_hist), "Non-finite values in controls"

        print(f"states len: {[len(s) for s in states_hist]}")
        print(f"controls len: {[len(u) for u in controls_hist]}")

    print("\nAll integration calls completed successfully.")


if __name__ == "__main__":
    main()
