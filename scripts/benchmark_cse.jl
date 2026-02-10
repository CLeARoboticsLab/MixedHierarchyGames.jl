#=
    Benchmark: CSE (Common Subexpression Elimination) impact on solver performance.

    Measures construction time, per-solve evaluation time, and memory
    for the LQ 3-player chain and PPV experiments, with and without CSE.

    Usage: julia --project=. scripts/benchmark_cse.jl
=#

using MixedHierarchyGames
using TrajectoryGamesBase: unflatten_trajectory
using Graphs: SimpleDiGraph, add_edge!
using TimerOutputs
using Printf

# ─── LQ Three-Player Chain Problem ──────────────────────────────────────────

function build_lq_three_player_chain(; cse::Bool=false)
    N = 3
    state_dim = 2
    control_dim = 2
    T = 3
    dt = 0.5

    G = SimpleDiGraph(N)
    add_edge!(G, 2, 1)
    add_edge!(G, 2, 3)

    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = fill(primal_dim, N)
    θs = setup_problem_parameter_variables(fill(state_dim, N))

    terminal_weight = 0.5
    control_weight = 0.05

    function J1(z1, z2, z3; θ=nothing)
        (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
        xs1, us1 = xs, us
        (; xs,) = unflatten_trajectory(z2, state_dim, control_dim)
        xs2 = xs
        terminal_weight * sum((xs1[end] .- xs2[end]) .^ 2) + control_weight * sum(sum(u .^ 2) for u in us1)
    end

    function J2(z1, z2, z3; θ=nothing)
        (; xs,) = unflatten_trajectory(z3, state_dim, control_dim)
        xs3 = xs
        (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
        xs2, us2 = xs, us
        (; xs,) = unflatten_trajectory(z1, state_dim, control_dim)
        xs1 = xs
        sum((0.5 * (xs1[end] .+ xs3[end])) .^ 2) + control_weight * sum(sum(u .^ 2) for u in us2)
    end

    function J3(z1, z2, z3; θ=nothing)
        (; xs, us) = unflatten_trajectory(z3, state_dim, control_dim)
        xs3, us3 = xs, us
        (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
        xs2, us2 = xs, us
        terminal_weight * sum((xs3[end] .- xs2[end]) .^ 2) +
            control_weight * sum(sum(u .^ 2) for u in us3) +
            control_weight * sum(sum(u .^ 2) for u in us2)
    end

    Js = Dict{Int,Any}(1 => J1, 2 => J2, 3 => J3)

    function make_constraints(i)
        return function (zi)
            dyn = mapreduce(vcat, 1:T) do t
                (; xs, us) = unflatten_trajectory(zi, state_dim, control_dim)
                xs[t+1] - xs[t] - dt .* us[t]
            end
            (; xs,) = unflatten_trajectory(zi, state_dim, control_dim)
            ic = xs[1] - θs[i]
            vcat(dyn, ic)
        end
    end
    gs = [make_constraints(i) for i in 1:N]

    x0 = Dict(1 => [0.0, 2.0], 2 => [2.0, 4.0], 3 => [6.0, 8.0])

    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, x0, name="LQ 3-player chain")
end

# ─── Pursuer-Protector-VIP Problem ──────────────────────────────────────────

function build_ppv(; cse::Bool=false)
    N = 3
    state_dim = 2
    control_dim = 2
    T = 20
    dt = 0.1

    G = SimpleDiGraph(N)
    add_edge!(G, 2, 1)
    add_edge!(G, 2, 3)

    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = fill(primal_dim, N)
    θs = setup_problem_parameter_variables(fill(state_dim, N))

    x_goal = [0.0, 0.0]

    function J1(z1, z2, z3; θ=nothing)
        (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
        xs1, us1 = xs, us
        (; xs,) = unflatten_trajectory(z2, state_dim, control_dim)
        xs2 = xs
        (; xs,) = unflatten_trajectory(z3, state_dim, control_dim)
        xs3 = xs
        chase = 2.0 * sum(sum((xs3[t] - xs1[t]) .^ 2) for t in 1:T)
        avoid = -1.0 * sum(sum((xs2[t] - xs1[t]) .^ 2) for t in 1:T)
        ctrl = 1.25 * sum(sum(u .^ 2) for u in us1)
        chase + avoid + ctrl
    end

    function J2(z1, z2, z3; θ=nothing)
        (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
        xs2, us2 = xs, us
        (; xs,) = unflatten_trajectory(z1, state_dim, control_dim)
        xs1 = xs
        (; xs,) = unflatten_trajectory(z3, state_dim, control_dim)
        xs3 = xs
        stay = 0.5 * sum(sum((xs3[t] - xs2[t]) .^ 2) for t in 1:T)
        protect = -1.0 * sum(sum((xs3[t] - xs1[t]) .^ 2) for t in 1:T)
        ctrl = 0.25 * sum(sum(u .^ 2) for u in us2)
        stay + protect + ctrl
    end

    function J3(z1, z2, z3; θ=nothing)
        (; xs, us) = unflatten_trajectory(z3, state_dim, control_dim)
        xs3, us3 = xs, us
        (; xs,) = unflatten_trajectory(z2, state_dim, control_dim)
        xs2 = xs
        goal = 10.0 * sum((xs3[end] .- x_goal) .^ 2)
        stay = 1.0 * sum(sum((xs3[t] - xs2[t]) .^ 2) for t in 1:T)
        ctrl = 1.25 * sum(sum(u .^ 2) for u in us3)
        goal + stay + ctrl
    end

    Js = Dict{Int,Any}(1 => J1, 2 => J2, 3 => J3)

    function make_constraints(i)
        return function (zi)
            dyn = mapreduce(vcat, 1:T) do t
                (; xs, us) = unflatten_trajectory(zi, state_dim, control_dim)
                xs[t+1] - xs[t] - dt .* us[t]
            end
            (; xs,) = unflatten_trajectory(zi, state_dim, control_dim)
            ic = xs[1] - θs[i]
            vcat(dyn, ic)
        end
    end
    gs = [make_constraints(i) for i in 1:N]

    x0 = Dict(1 => [-5.0, 1.0], 2 => [-2.0, -2.5], 3 => [2.0, -4.0])

    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, x0, name="PPV")
end

# ─── Benchmark Runner ──────────────────────────────────────────────────────

function benchmark_problem(problem_builder; n_solves=3)
    results = Dict{String, Any}()

    for cse_enabled in [false, true]
        label = cse_enabled ? "with_cse" : "without_cse"
        prob = problem_builder(; cse=cse_enabled)

        # --- Construction benchmark ---
        construction_alloc = @allocated begin
            construction_time = @elapsed begin
                solver = NonlinearSolver(
                    prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                    prob.state_dim, prob.control_dim;
                    cse=cse_enabled, max_iters=50, tol=1e-8
                )
            end
        end

        # --- Solve benchmark (warmup + measured) ---
        # Warmup solve
        result_warmup = solve_raw(solver, prob.x0)

        # Measured solves
        solve_times = Float64[]
        solve_allocs = Int[]
        for _ in 1:n_solves
            alloc = @allocated begin
                t = @elapsed begin
                    result = solve_raw(solver, prob.x0)
                end
            end
            push!(solve_times, t)
            push!(solve_allocs, alloc)
        end

        results[label] = (;
            construction_time,
            construction_alloc,
            solve_times,
            solve_allocs,
            converged=result_warmup.converged,
            iterations=result_warmup.iterations,
            residual=result_warmup.residual,
        )
    end

    return results
end

function format_bytes(bytes)
    if bytes < 1024
        return @sprintf("%d B", bytes)
    elseif bytes < 1024^2
        return @sprintf("%.1f KiB", bytes / 1024)
    elseif bytes < 1024^3
        return @sprintf("%.1f MiB", bytes / 1024^2)
    else
        return @sprintf("%.1f GiB", bytes / 1024^3)
    end
end

function format_time(seconds)
    if seconds < 1.0
        return @sprintf("%.1f ms", seconds * 1000)
    else
        return @sprintf("%.2f s", seconds)
    end
end

function print_results(name, results)
    no_cse = results["without_cse"]
    with_cse = results["with_cse"]

    avg_solve_no_cse = sum(no_cse.solve_times) / length(no_cse.solve_times)
    avg_solve_cse = sum(with_cse.solve_times) / length(with_cse.solve_times)
    avg_alloc_no_cse = sum(no_cse.solve_allocs) / length(no_cse.solve_allocs)
    avg_alloc_cse = sum(with_cse.solve_allocs) / length(with_cse.solve_allocs)

    construct_change = (with_cse.construction_time - no_cse.construction_time) / no_cse.construction_time * 100
    solve_change = (avg_solve_cse - avg_solve_no_cse) / avg_solve_no_cse * 100
    construct_mem_change = (with_cse.construction_alloc - no_cse.construction_alloc) / no_cse.construction_alloc * 100

    println("\n## $name")
    println()
    println("| Metric | Without CSE | With CSE | Change |")
    println("|--------|-------------|----------|--------|")
    @printf("| Construction time | %s | %s | %+.1f%% |\n",
        format_time(no_cse.construction_time), format_time(with_cse.construction_time), construct_change)
    @printf("| Construction memory | %s | %s | %+.1f%% |\n",
        format_bytes(no_cse.construction_alloc), format_bytes(with_cse.construction_alloc), construct_mem_change)
    @printf("| Per-solve time (avg) | %s | %s | %+.1f%% |\n",
        format_time(avg_solve_no_cse), format_time(avg_solve_cse), solve_change)
    @printf("| Per-solve memory (avg) | %s | %s | %+.1f%% |\n",
        format_bytes(round(Int, avg_alloc_no_cse)), format_bytes(round(Int, avg_alloc_cse)),
        (avg_alloc_cse - avg_alloc_no_cse) / avg_alloc_no_cse * 100)
    println("| Converged | $(no_cse.converged) | $(with_cse.converged) | - |")
    println("| Iterations | $(no_cse.iterations) | $(with_cse.iterations) | - |")
    @printf("| Final residual | %.2e | %.2e | - |\n", no_cse.residual, with_cse.residual)
end

# ─── Main ───────────────────────────────────────────────────────────────────

function main()
    println("# CSE Benchmark Results")
    println()
    println("Benchmarking Common Subexpression Elimination impact on solver performance.")
    println("Each solve is run $(3) times after warmup.")

    println("\nRunning LQ 3-player chain benchmark...")
    lq_results = benchmark_problem(build_lq_three_player_chain; n_solves=3)
    print_results("LQ 3-Player Chain", lq_results)

    println("\nRunning PPV benchmark...")
    ppv_results = benchmark_problem(build_ppv; n_solves=3)
    print_results("Pursuer-Protector-VIP (PPV)", ppv_results)
end

main()
