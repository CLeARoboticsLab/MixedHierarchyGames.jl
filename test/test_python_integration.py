#!/usr/bin/env python3
"""
Minimal Python integration test for the Julia solver.

This script demonstrates how to call the Julia solver from Python
and run it repeatedly in a loop.

Requirements:
    pip install julia

Before first use, you need to install PyCall in Julia:
    python3 -c "import julia; julia.install()"
"""

from julia import Main
import numpy as np
from pathlib import Path

def main():
    print("Loading Julia solver...")

    # Activate and instantiate the project to ensure all deps are available
    project_root = str(Path(__file__).resolve().parents[1])
    Main.eval(
        f"""
        import Pkg
        Pkg.activate(raw"{project_root}")
        try
            Pkg.instantiate()
        catch e
            @warn "Pkg.instantiate() failed" exception=e
        end
        """
    )

    # Include your Julia file (from the project after activation)
    Main.eval(f"include(raw\"{project_root}/legacy/test_automatic_solver.jl\")")
    
    print("Precompiling solver with initial run...")
    # Precompile the solver once (optional but recommended for performance)
    Main.eval("""
    x0_initial = [[0.0, 2.0], [2.0, 4.0], [6.0, 8.0]]
    nplayer_hierarchy_navigation(x0_initial; verbose=false)
    """)
    print("Precompilation complete.\n")
    
    # Test 1: Single call
    print("=" * 60)
    print("Test 1: Single solver call")
    print("=" * 60)
    x0 = [[0.0, 2.0], [2.0, 4.0], [6.0, 8.0]]
    print(f"Initial conditions: {x0}")
    
    result = Main.nplayer_hierarchy_navigation(x0)
    next_state, curr_control = result
    
    print(f"Next state: {next_state}")
    print(f"Current control: {curr_control}")
    
    # Test 2: Multiple calls with varying initial conditions
    print("\n" + "=" * 60)
    print("Test 2: Multiple solver calls in a loop")
    print("=" * 60)
    
    x_current = [[0.0, 2.0], [2.0, 4.0], [6.0, 8.0]]
    
    for step in range(5):
        print(f"\n--- Step {step} ---")
        print(f"Current state: {x_current}")
        
        # Call the solver
        result = Main.nplayer_hierarchy_navigation(x_current)
        next_state, curr_control = result
        
        print(f"Control: {curr_control}")
        print(f"Next state: {next_state}")
        
        # Update current state for next iteration
        # Convert to Python lists if needed
        x_current = [[float(x) for x in state] for state in next_state]
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
