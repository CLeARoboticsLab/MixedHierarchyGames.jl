# Test tier classification for CI optimization.
# Fast tests: unit tests, utilities, validation, small solves (< 2 min total)
# Slow tests: nonlinear solver convergence, integration tests, OLSE nonlinear (minutes)

const FAST_TEST_FILES = [
    "test_graph_utils.jl",
    "test_ordered_player_indices.jl",
    "test_symbolic_utils.jl",
    "test_problem_setup.jl",
    "test_qp_kkt.jl",
    "test_linesearch.jl",
    "test_input_validation.jl",
    "test_type_stability.jl",
    "test_qp_solver.jl",
    "test_qp_failure_modes.jl",
    "test_interface.jl",
    "test_timer.jl",
    "test_type_bounds.jl",
    "test_block_arrays.jl",
    "test_unified_interface.jl",
    "olse/test_qp_solver.jl",
]

const SLOW_TEST_FILES = [
    "test_nonlinear_solver.jl",
    "test_kkt_verification.jl",
    "test_integration.jl",
    "test_flexible_callsite.jl",
    "olse/test_nonlinear_solver.jl",
    "test_sparse_solve.jl",
    "test_allocation_optimization.jl",
]

"""
    get_test_files(fast_only::Bool) -> Vector{String}

Return the list of test files to run based on tier selection.
When `fast_only` is true, only fast-tier tests are returned.
When false, all tests (fast + slow) are returned.
"""
function get_test_files(fast_only::Bool)
    if fast_only
        return copy(FAST_TEST_FILES)
    else
        return vcat(FAST_TEST_FILES, SLOW_TEST_FILES)
    end
end
