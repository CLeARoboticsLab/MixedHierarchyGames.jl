using Test
using Graphs: SimpleDiGraph, add_edge!, nv
using MixedHierarchyGames:
    setup_approximate_kkt_solver,
    setup_problem_variables,
    setup_problem_parameter_variables,
    default_backend

# Reuse shared test helpers
# (testing_utils.jl is already included by runtests.jl)

@testset "K symbol flattening deduplication" begin
    # Build a standard 2-player problem
    prob = make_standard_two_player_problem(; T=3, state_dim=2, control_dim=2)
    G = prob.G
    N = prob.N

    backend = default_backend()
    problem_vars = setup_problem_variables(G, prob.primal_dims, prob.gs; backend)
    all_variables = problem_vars.all_variables
    zs = problem_vars.zs
    λs = problem_vars.λs
    μs = problem_vars.μs
    ws = problem_vars.ws
    ys = problem_vars.ys

    all_augmented_variables, setup_info = setup_approximate_kkt_solver(
        G, prob.Js, zs, λs, μs, prob.gs, ws, ys, prob.θs,
        all_variables, backend;
        verbose=false, cse=false
    )

    K_syms = setup_info.K_syms

    # Compute all_K_syms_vec the old way (inline)
    all_K_syms_vec_inline = vcat([reshape(something(K_syms[ii], eltype(all_variables)[]), :) for ii in 1:N]...)

    # The setup_info should now include all_K_syms_vec directly
    @test haskey(pairs(setup_info), :all_K_syms_vec)

    # The returned all_K_syms_vec should be identical to the inline computation
    # Use isequal for symbolic comparison (== returns symbolic Num, not Bool)
    @test length(setup_info.all_K_syms_vec) == length(all_K_syms_vec_inline)
    @test all(isequal.(setup_info.all_K_syms_vec, all_K_syms_vec_inline))

    # Verify all_augmented_variables is consistent: should equal vcat(all_variables, all_K_syms_vec)
    expected_augmented = vcat(all_variables, setup_info.all_K_syms_vec)
    @test length(all_augmented_variables) == length(expected_augmented)
    @test all(isequal.(all_augmented_variables, expected_augmented))
end
