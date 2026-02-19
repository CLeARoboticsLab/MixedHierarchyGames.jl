#=
    Benchmark: quadratic interpolation linesearch vs geometric/armijo

    Compares solve time, iterations, and linesearch statistics across
    available linesearch methods.

    Usage: julia --project=. scripts/benchmark_quadratic_linesearch.jl
=#

using MixedHierarchyGames
using MixedHierarchyGames: setup_problem_parameter_variables

using Graphs: SimpleDiGraph, add_edge!
using TrajectoryGamesBase: unflatten_trajectory
using Statistics: median, mean
using LinearAlgebra: norm

# ── Problem builders ────────────────────────────────────────────────────

function make_three_player_chain(; T=3, state_dim=2, control_dim=2)
    N = 3
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 3)
    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)
    θs = setup_problem_parameter_variables(fill(state_dim, N))
    function make_cost(player_idx, goal)
        function cost(zs...; θ=nothing)
            z = zs[player_idx]
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        return cost
    end
    goals = [[Float64(i), Float64(i)] for i in 1:N]
    Js = Dict(i => make_cost(i, goals[i]) for i in 1:N)
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

function make_four_player_chain(; T=5, state_dim=4, control_dim=2)
    N = 4
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 3)
    add_edge!(G, 3, 4)
    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)
    θs = setup_problem_parameter_variables(fill(state_dim, N))
    function make_cost(player_idx, goal)
        function cost(zs...; θ=nothing)
            z = zs[player_idx]
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        return cost
    end
    goals = [[Float64(i), Float64(i), 0.0, 0.0] for i in 1:N]
    Js = Dict(i => make_cost(i, goals[i]) for i in 1:N)
    function make_dynamics(player_idx)
        function dynamics_constraint(z)
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            constraints = []
            for t in 1:T
                x_next = copy(xs[t])
                x_next[1] += us[t][1]
                x_next[2] += us[t][2]
                push!(constraints, xs[t+1] - x_next)
            end
            push!(constraints, xs[1] - θs[player_idx])
            return vcat(constraints...)
        end
        return dynamics_constraint
    end
    gs = [make_dynamics(i) for i in 1:N]
    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, T, N)
end

# ── Benchmark helpers ───────────────────────────────────────────────────

function benchmark_solve(prob; n_solves=20, linesearch_method=:geometric)
    solver = NonlinearSolver(
        prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
        prob.state_dim, prob.control_dim;
        linesearch_method
    )
    params = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

    # Warmup
    solve_raw(solver, params; max_iters=5)

    times = Float64[]
    allocs = Int[]
    iterations_list = Int[]
    step_sizes_all = Vector{Float64}[]

    for _ in 1:n_solves
        step_sizes = Float64[]
        callback = info -> push!(step_sizes, info.step_size)

        a = @allocated t = @elapsed begin
            result = solve_raw(solver, params; callback=callback)
        end
        push!(times, t)
        push!(allocs, a)
        push!(iterations_list, result.iterations)
        push!(step_sizes_all, step_sizes)
    end

    # Compute linesearch statistics
    all_backtracks = Float64[]
    full_step_count = 0
    total_steps = 0
    for step_sizes in step_sizes_all
        for α in step_sizes
            total_steps += 1
            if α == 1.0
                full_step_count += 1
                push!(all_backtracks, 0.0)
            elseif α > 0.0
                push!(all_backtracks, log(α) / log(0.5))
            end
        end
    end

    return (;
        median_time=median(times),
        median_alloc=median(allocs),
        median_iters=median(iterations_list),
        mean_backtracks=isempty(all_backtracks) ? NaN : mean(all_backtracks),
        full_step_rate=total_steps > 0 ? full_step_count / total_steps : NaN,
    )
end

# ── Main ────────────────────────────────────────────────────────────────

println("=" ^ 70)
println("Quadratic Interpolation Linesearch Benchmark")
println("=" ^ 70)
println()

problems = [
    ("3-player chain (T=3, s=2, c=2)", make_three_player_chain(), 100),
    ("4-player chain (T=5, s=4, c=2)", make_four_player_chain(), 20),
]

methods = [:geometric, :armijo, :armijo_quadratic]

for (label, prob, n_solves) in problems
    println("── $label ($n_solves runs) ──")

    for method in methods
        result = benchmark_solve(prob; n_solves, linesearch_method=method)
        t_ms = round(result.median_time * 1000, digits=2)
        a_kb = round(result.median_alloc / 1024, digits=1)
        iters = round(Int, result.median_iters)
        backtracks = round(result.mean_backtracks, digits=2)
        full_rate = round(result.full_step_rate * 100, digits=1)

        println("  :$method")
        println("    Time: $(t_ms)ms | Iters: $iters | Alloc: $(a_kb)KB")
        println("    Mean backtracks/iter: $backtracks | Full-step rate: $(full_rate)%")
    end
    println()
end

println("=" ^ 70)
println("Done.")
