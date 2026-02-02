using Test
using MixedHierarchyGames

@testset "MixedHierarchyGames.jl" begin
    # Phase A: Utilities (implemented)
    include("test_graph_utils.jl")
    include("test_symbolic_utils.jl")

    # Phase B: Problem Setup (TDD - tests first)
    include("test_problem_setup.jl")

    # Phase C: QP KKT Construction (TDD)
    include("test_qp_kkt.jl")

    # Phase E: Linesearch (TDD)
    include("test_linesearch.jl")

    # Phase F: Nonlinear Solver (TDD)
    include("test_nonlinear_solver.jl")

    # Phase G: Interface (TDD)
    include("test_interface.jl")
end
