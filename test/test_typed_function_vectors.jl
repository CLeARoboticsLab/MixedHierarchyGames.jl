using Test
using Graphs: SimpleDiGraph, add_edge!
using Symbolics: Num
using MixedHierarchyGames:
    NonlinearSolver,
    NonlinearSolverOptions,
    MNFunctionWrapper,
    setup_problem_variables,
    setup_problem_parameter_variables,
    setup_approximate_kkt_solver,
    compute_K_evals,
    unflatten_trajectory,
    default_backend

@testset "Typed Function Vectors (Perf #6)" begin
    # Setup a 2-player hierarchy for testing
    G = SimpleDiGraph(2)
    add_edge!(G, 1, 2)
    primal_dims = [2, 2]
    backend = default_backend()
    gs = [z -> Num[] for _ in 1:2]

    θs = setup_problem_parameter_variables([2, 2]; backend)
    Js = Dict(
        1 => (z1, z2; θ=nothing) -> sum(z1.^2) + sum(z2.^2),
        2 => (z1, z2; θ=nothing) -> sum(z2.^2)
    )

    problem_vars = setup_problem_variables(G, primal_dims, gs; backend)
    zs = problem_vars.zs
    λs = problem_vars.λs
    μs = problem_vars.μs
    ws = problem_vars.ws
    ys = problem_vars.ys
    all_variables = problem_vars.all_variables

    _, setup_info = setup_approximate_kkt_solver(
        G, Js, zs, λs, μs, gs, ws, ys, θs, all_variables, backend
    )

    @testset "M_fns!/N_fns! element type is concrete (not abstract Function)" begin
        M_fns = setup_info.var"M_fns!"
        N_fns = setup_info.var"N_fns!"

        # The element type should NOT be the abstract `Function` type.
        # It should be MNFunctionWrapper (a concrete callable type).
        @test eltype(M_fns) !== Function
        @test eltype(N_fns) !== Function

        # The element type should be a concrete type
        @test isconcretetype(eltype(M_fns))
        @test isconcretetype(eltype(N_fns))

        # Specifically, it should be MNFunctionWrapper
        @test eltype(M_fns) === MNFunctionWrapper
        @test eltype(N_fns) === MNFunctionWrapper
    end

    @testset "M_fn/N_fn calls are type-stable from Vector indexing" begin
        # When we index into the vector and call the function, the return type
        # should be inferrable, not Any.
        # The wrapped functions return Nothing (buffer is mutated in-place).
        M_fns = setup_info.var"M_fns!"
        N_fns = setup_info.var"N_fns!"

        # Player 2 is the follower (has a leader), so it has real M_fn/N_fn
        ii = 2
        π_size = setup_info.π_sizes[ii]
        M_buf = Matrix{Float64}(undef, π_size, length(ws[ii]))
        N_buf = Matrix{Float64}(undef, π_size, length(ys[ii]))
        z_test = zeros(length(all_variables))

        # The call through the vector should be inferrable
        function call_M_fn!(M_fns, ii, buf, z)
            M_fns[ii](buf, z)
        end
        function call_N_fn!(N_fns, ii, buf, z)
            N_fns[ii](buf, z)
        end

        # @inferred verifies that the return type is Nothing, not Any
        @test @inferred(call_M_fn!(M_fns, ii, M_buf, z_test)) === nothing
        @test @inferred(call_N_fn!(N_fns, ii, N_buf, z_test)) === nothing
    end

    @testset "compute_K_evals correctness with typed function vectors" begin
        z_current = zeros(length(all_variables))

        # compute_K_evals should work correctly with typed function vectors
        all_K_vec, info = compute_K_evals(z_current, problem_vars, setup_info)

        # Results should still be correct
        @test info.status == :ok
        @test all_K_vec isa Vector{Float64}

        # K_evals for player 2 (follower) should be a matrix
        @test info.K_evals[2] isa Matrix{Float64}
        # K_evals for player 1 (root) should be nothing
        @test info.K_evals[1] === nothing
    end
end
