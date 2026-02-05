#=
    Convergence Analysis - Configuration

    Pure parameters for multi-run convergence testing.
=#

# Default analysis configuration
const DEFAULT_CONFIG = (
    R = 6.0,                    # turning radius (passed to lane change)
    T = 14,                     # time horizon
    Δt = 0.4,                   # time step
    num_runs = 11,              # number of runs with perturbed initial states
    max_iters = 200,            # max solver iterations per run
    perturb_scale = 0.1,        # perturbation scale for initial states
    seed = 1234,                # random seed for reproducibility
)

# Solver tolerance
const TOLERANCE = 1e-6
