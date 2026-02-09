using Test
using MixedHierarchyGames

# Include shared testing utilities
include("testing_utils.jl")

@testset "MixedHierarchyGames.jl" begin
    # Phase A: Utilities (implemented)
    include("test_graph_utils.jl")
    include("test_symbolic_utils.jl")

    # Phase B: Problem Setup (TDD - tests first)
    include("test_problem_setup.jl")

    # Phase C: QP KKT Construction (TDD)
    include("test_qp_kkt.jl")

    # Phase D: QP Solver (TDD)
    include("test_qp_solver.jl")

    # Phase E: Linesearch (TDD)
    include("test_linesearch.jl")

    # Phase F: Nonlinear Solver (TDD)
    include("test_nonlinear_solver.jl")

    # KKT Verification Utilities (TDD)
    include("test_kkt_verification.jl")

    # Phase G: Interface (TDD)
    include("test_interface.jl")

    # Input Validation
    include("test_input_validation.jl")

    # Integration Tests (QP vs Nonlinear, paper examples)
    include("test_integration.jl")

    # OLSE Tests (comparing solvers against closed-form OLSE solution)
    include("olse/test_qp_solver.jl")
    include("olse/test_nonlinear_solver.jl")

    # Type Stability Tests
    include("test_type_stability.jl")

    # TimerOutputs Integration
    include("test_timer.jl")

    # Pre-allocation Tests
    include("test_preallocation.jl")
end
