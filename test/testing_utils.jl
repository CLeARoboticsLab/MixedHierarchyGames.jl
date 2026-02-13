#=
    Shared test utilities for MixedHierarchyGames test suite.
=#

using MixedHierarchyGames: make_symbolic_vector, setup_problem_parameter_variables
using Graphs: SimpleDiGraph, add_edge!
using TrajectoryGamesBase: unflatten_trajectory

"""
    make_θ(player::Int, dim::Int)

Convenience helper to create parameter vectors using SymbolicTracingUtils.
Equivalent to `make_symbolic_vector(:θ, player, dim)`.

Used in tests to create symbolic parameter variables without calling Symbolics directly.
"""
make_θ(player::Int, dim::Int) = make_symbolic_vector(:θ, player, dim)

"""
    make_standard_two_player_problem(; goal1=[1.0, 1.0], goal2=[2.0, 2.0],
                                       T=3, state_dim=2, control_dim=2)

Build a standard 2-player leader-follower NonlinearSolver problem for tests.

Returns a NamedTuple with: `G`, `Js`, `gs`, `primal_dims`, `θs`, `state_dim`,
`control_dim`, `T`, `N`.

The problem uses integrator dynamics (x[t+1] = x[t] + u[t]) with initial
condition x[1] = θ[player], and quadratic terminal-cost + control-effort objectives.
"""
function make_standard_two_player_problem(;
    goal1=[1.0, 1.0],
    goal2=[2.0, 2.0],
    T=3,
    state_dim=2,
    control_dim=2,
)
    N = 2
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)

    primal_dim_per_player = (state_dim + control_dim) * (T + 1)
    primal_dims = fill(primal_dim_per_player, N)

    θs = setup_problem_parameter_variables(fill(state_dim, N))

    function J1(z1, z2; θ=nothing)
        (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
        sum((xs[end] .- goal1) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    function J2(z1, z2; θ=nothing)
        (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
        sum((xs[end] .- goal2) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    Js = Dict(1 => J1, 2 => J2)

    function make_dynamics(player_idx)
        function dynamics_constraint(z)
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            constraints = []
            for t in 1:T
                push!(constraints, xs[t+1] - xs[t] - us[t])
            end
            push!(constraints, xs[1] - θs[player_idx])
            return vcat(constraints...)
        end
        return dynamics_constraint
    end

    gs = [make_dynamics(i) for i in 1:N]
    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, T, N)
end

"""
    make_simple_qp_two_player()

Build a simple 2-player QP problem with state_dim=1, control_dim=1.

Returns a NamedTuple with: `G`, `Js`, `gs`, `primal_dims`, `θs`, `state_dim`,
`control_dim`, `θ1_vec`, `θ2_vec`.

Used for testing parameter passing interfaces (Dict vs Vector-of-Vectors).
"""
function make_simple_qp_two_player()
    G = SimpleDiGraph(2)
    add_edge!(G, 1, 2)

    primal_dims = [4, 4]
    state_dim = 1
    control_dim = 1

    θ1_vec = make_θ(1, 1)
    θ2_vec = make_θ(2, 1)
    θs = Dict(1 => θ1_vec, 2 => θ2_vec)

    gs = [
        z -> [z[1] - θ1_vec[1]],
        z -> [z[1] - θ2_vec[1]],
    ]

    Js = Dict(
        1 => (z1, z2; θ=nothing) -> sum(z1 .^ 2),
        2 => (z1, z2; θ=nothing) -> sum(z2 .^ 2),
    )

    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, θ1_vec, θ2_vec)
end
