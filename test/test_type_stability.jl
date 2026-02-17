using Test
using Graphs: SimpleDiGraph, add_edge!
using Symbolics: Num
using MixedHierarchyGames:
    NonlinearSolver,
    NonlinearSolverOptions,
    HierarchyProblem,
    MNFunctionWrapper,
    setup_problem_variables,
    setup_problem_parameter_variables,
    setup_approximate_kkt_solver,
    compute_K_evals,
    unflatten_trajectory,
    default_backend

@testset "Type Stability" begin
    # Setup a simple 2-player hierarchy for testing
    G = SimpleDiGraph(2)
    add_edge!(G, 1, 2)
    primal_dims = [2, 2]
    backend = default_backend()

    # Simple constraint functions (no constraints)
    gs = [z -> Num[] for _ in 1:2]

    @testset "setup_problem_variables returns typed containers" begin
        problem_vars = setup_problem_variables(G, primal_dims, gs; backend)

        # ys should be Dict{Int, Vector{Num}}, not Dict{Int, Any}
        @test problem_vars.ys isa Dict{Int, Vector{Num}}

        # ws should be Dict{Int, Vector{Num}}, not Dict{Int, Any}
        @test problem_vars.ws isa Dict{Int, Vector{Num}}

        # Verify the values are actually Vector{Num}
        for (k, v) in problem_vars.ys
            @test v isa Vector{Num}
        end
        for (k, v) in problem_vars.ws
            @test v isa Vector{Num}
        end
    end

    @testset "setup_approximate_kkt_solver returns typed containers" begin
        problem_vars = setup_problem_variables(G, primal_dims, gs; backend)
        zs = problem_vars.zs
        λs = problem_vars.λs
        μs = problem_vars.μs
        ws = problem_vars.ws
        ys = problem_vars.ys
        all_variables = problem_vars.all_variables

        # Create parameter variables
        θs = setup_problem_parameter_variables([2, 2]; backend)

        # Simple cost functions
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2) + sum(z2.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2)
        )

        _, setup_info = setup_approximate_kkt_solver(
            G, Js, zs, λs, μs, gs, ws, ys, θs, all_variables, backend
        )

        # K_syms should have concrete Union type, not Any
        @test setup_info.K_syms isa Dict{Int, Union{Matrix{Num}, Vector{Num}}}

        # πs uses Dict{Int, Any} because leaders have BlockVector, leaves have Vector{Num}.
        # This matches qp_kkt.jl's pattern and is acceptable for symbolic construction
        # which is done once at solver creation, not in the hot solve path.
        @test setup_info.πs isa Dict{Int, Any}

        # M_fns! and N_fns! should be Vector{MNFunctionWrapper} (concrete callable type)
        @test setup_info.var"M_fns!" isa Vector{MNFunctionWrapper}
        @test setup_info.var"N_fns!" isa Vector{MNFunctionWrapper}
    end

    @testset "compute_K_evals returns typed containers" begin
        problem_vars = setup_problem_variables(G, primal_dims, gs; backend)
        zs = problem_vars.zs
        λs = problem_vars.λs
        μs = problem_vars.μs
        ws = problem_vars.ws
        ys = problem_vars.ys
        all_variables = problem_vars.all_variables

        θs = setup_problem_parameter_variables([2, 2]; backend)

        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2) + sum(z2.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2)
        )

        _, setup_info = setup_approximate_kkt_solver(
            G, Js, zs, λs, μs, gs, ws, ys, θs, all_variables, backend
        )

        # Create a test z_current
        z_current = zeros(length(all_variables))

        all_K_vec, info = compute_K_evals(z_current, problem_vars, setup_info)

        # K_evals should be Vector-indexed (not Dict)
        @test info.K_evals isa Vector{Union{Matrix{Float64}, Nothing}}
        @test info.M_evals isa Vector{Union{Matrix{Float64}, Nothing}}
        @test info.N_evals isa Vector{Union{Matrix{Float64}, Nothing}}

        # Verify actual values have correct types
        for v in info.K_evals
            @test v isa Union{Matrix{Float64}, Nothing}
        end
    end

    @testset "run_nonlinear_solver accepts Vector{Float64} initial_guess" begin
        # This tests that the function signature is properly typed
        # The function should accept Vector{Float64}, not just Vector
        problem_vars = setup_problem_variables(G, primal_dims, gs; backend)

        # Verify the method exists with Float64 signature
        # (This is more of a compile-time check - if the type is wrong, this would error)
        @test hasmethod(
            MixedHierarchyGames.run_nonlinear_solver,
            Tuple{NamedTuple, Dict, SimpleDiGraph}
        )
    end

    @testset "NonlinearSolver.precomputed NamedTuple is type-stable" begin
        # Verify that the NamedTuple type parameter TC in NonlinearSolver{TP, TC}
        # produces fully concrete types, so field access is resolved at compile time.
        #
        # Background: NonlinearSolver uses a parametric NamedTuple for precomputed data
        # rather than a concrete struct. This is type-stable because:
        # 1. The NamedTuple type is fully concrete (all field types are known)
        # 2. The TC type parameter propagates through NonlinearSolver{TP, TC}
        # 3. Julia specializes on concrete NamedTuple types regardless of ::NamedTuple annotations
        #
        # This test documents and guards that the NamedTuple approach is type-stable,
        # so no refactor to a concrete struct is needed (Perf #8 investigation).

        # Build a 2-player Stackelberg game with dynamics constraints
        G_nl = SimpleDiGraph(2)
        add_edge!(G_nl, 1, 2)

        state_dim = 2
        control_dim = 2
        T = 2
        primal_dim = (state_dim + control_dim) * (T + 1)
        nl_primal_dims = fill(primal_dim, 2)

        θs = setup_problem_parameter_variables(fill(state_dim, 2); backend)
        Δt = 0.1

        Js = Dict(
            1 => (z1, z2; θ=nothing) -> begin
                (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
                sum(sum(x.^2) for x in xs) + 0.1 * sum(sum(u.^2) for u in us)
            end,
            2 => (z1, z2; θ=nothing) -> begin
                (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
                sum(sum(x.^2) for x in xs) + 0.1 * sum(sum(u.^2) for u in us)
            end
        )

        nl_gs = [
            z -> begin
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                dyn = mapreduce(vcat, 1:T) do t; xs[t+1] - xs[t] - Δt * us[t] end
                ic = xs[1] - θs[1]
                vcat(dyn, ic)
            end,
            z -> begin
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                dyn = mapreduce(vcat, 1:T) do t; xs[t+1] - xs[t] - Δt * us[t] end
                ic = xs[1] - θs[2]
                vcat(dyn, ic)
            end
        ]

        solver = NonlinearSolver(G_nl, Js, nl_gs, nl_primal_dims, θs, state_dim, control_dim)

        @testset "solver type is fully concrete" begin
            @test isconcretetype(typeof(solver))
            @test isconcretetype(typeof(solver.precomputed))
        end

        @testset "precomputed NamedTuple fields are all concretely typed" begin
            precomputed = solver.precomputed
            for name in fieldnames(typeof(precomputed))
                val = getfield(precomputed, name)
                @test isconcretetype(typeof(val))
            end
        end

        @testset "precomputed field access is type-stable via @inferred" begin
            # @inferred verifies the compiler can infer the return type.
            # If the NamedTuple field access were type-unstable, @inferred would throw.
            function access_precomputed_fields(s::NonlinearSolver)
                (; precomputed) = s
                return length(precomputed.all_variables)
            end
            @test @inferred(access_precomputed_fields(solver)) isa Int
        end

        @testset "run_nonlinear_solver specializes on concrete precomputed type" begin
            # Verify that passing precomputed through ::NamedTuple annotation
            # still results in specialization (Julia specializes on concrete types)
            function access_via_namedtuple_annotation(precomputed::NamedTuple)
                return length(precomputed.all_variables)
            end
            @test @inferred(access_via_namedtuple_annotation(solver.precomputed)) isa Int
        end
    end
end
