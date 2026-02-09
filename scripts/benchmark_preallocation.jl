#=
    Pre-allocation Benchmark Script

    Compares default vs preallocate=true on multiple problem sizes:
      1. LQ 3-player chain (small: STATE_DIM=2, T=10)
      2. Pursuer-Protector-VIP (medium: STATE_DIM=2, T=20)
      3. Nonlinear lane change (large: STATE_DIM=4, T=10) - optional, slow construction

    Usage:
      julia --project=experiments scripts/benchmark_preallocation.jl
      julia --project=experiments scripts/benchmark_preallocation.jl --lane-change  # include lane change
=#

using MixedHierarchyGames
using MixedHierarchyGames: run_nonlinear_solver, compute_K_evals, preoptimize_nonlinear_solver
using TimerOutputs: TimerOutput, TimerOutputs, @timeit
using TrajectoryGamesBase: unflatten_trajectory
using Graphs: SimpleDiGraph, add_edge!, nv
using LinearAlgebra: norm

const RUN_LANE_CHANGE = "--lane-change" in ARGS

# ═══════════════════════════════════════════════════════
# Problem 1: LQ 3-player chain (small)
# ═══════════════════════════════════════════════════════
function build_lq_chain(; T=10, state_dim=2, control_dim=2, Δt=0.5)
    N = 3
    G = SimpleDiGraph(N)
    add_edge!(G, 2, 1)
    add_edge!(G, 2, 3)

    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = fill(primal_dim, N)
    θs = setup_problem_parameter_variables(fill(state_dim, N))

    function J₁(z₁, z₂, z₃; θ=nothing)
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, us¹ = xs, us
        (; xs,) = unflatten_trajectory(z₂, state_dim, control_dim)
        0.5 * sum((xs¹[end] .- xs[end]) .^ 2) + 0.05 * sum(sum(u .^ 2) for u in us¹)
    end
    function J₂(z₁, z₂, z₃; θ=nothing)
        (; xs,) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³ = xs
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us
        (; xs,) = unflatten_trajectory(z₁, state_dim, control_dim)
        sum((0.5 * (xs[end] .+ xs³[end])) .^ 2) + 0.05 * sum(sum(u .^ 2) for u in us²)
    end
    function J₃(z₁, z₂, z₃; θ=nothing)
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, us³ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        0.5 * sum((xs³[end] .- xs[end]) .^ 2) + 0.05 * sum(sum(u³ .^ 2) for u³ in us³) + 0.05 * sum(sum(u² .^ 2) for u² in us)
    end
    Js = Dict{Int,Any}(1 => J₁, 2 => J₂, 3 => J₃)

    function make_constraints(i)
        return function (zᵢ)
            dyn = mapreduce(vcat, 1:T) do t
                (; xs, us) = unflatten_trajectory(zᵢ, state_dim, control_dim)
                xs[t+1] - xs[t] - Δt * us[t]
            end
            (; xs,) = unflatten_trajectory(zᵢ, state_dim, control_dim)
            ic = xs[1] - θs[i]
            vcat(dyn, ic)
        end
    end
    gs = [make_constraints(i) for i in 1:N]

    params = Dict(1 => [0.0, 2.0], 2 => [2.0, 4.0], 3 => [6.0, 8.0])
    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, params, N, T, name="LQ 3-player chain (T=$T)")
end

# ═══════════════════════════════════════════════════════
# Problem 2: Pursuer-Protector-VIP (medium)
# ═══════════════════════════════════════════════════════
function build_ppv(; T=20, state_dim=2, control_dim=2, Δt=0.1)
    N = 3
    G = SimpleDiGraph(N)
    add_edge!(G, 2, 1)
    add_edge!(G, 2, 3)

    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = fill(primal_dim, N)
    θs = setup_problem_parameter_variables(fill(state_dim, N))

    x_goal = [0.0, 0.0]

    function J₁(z₁, z₂, z₃; θ=nothing)
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, us¹ = xs, us
        (; xs,) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs² = xs
        (; xs,) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³ = xs
        chase = 2.0 * sum(sum((xs³[t] - xs¹[t]) .^ 2) for t in 1:T)
        avoid = -1.0 * sum(sum((xs²[t] - xs¹[t]) .^ 2) for t in 1:T)
        ctrl = 1.25 * sum(sum(u .^ 2) for u in us¹)
        chase + avoid + ctrl
    end
    function J₂(z₁, z₂, z₃; θ=nothing)
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us
        (; xs,) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹ = xs
        (; xs,) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³ = xs
        stay = 0.5 * sum(sum((xs³[t] - xs²[t]) .^ 2) for t in 1:T)
        protect = -1.0 * sum(sum((xs³[t] - xs¹[t]) .^ 2) for t in 1:T)
        ctrl = 0.25 * sum(sum(u .^ 2) for u in us²)
        stay + protect + ctrl
    end
    function J₃(z₁, z₂, z₃; θ=nothing)
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, us³ = xs, us
        (; xs,) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs² = xs
        goal = 10.0 * sum((xs³[end] .- x_goal) .^ 2)
        stay = 1.0 * sum(sum((xs³[t] - xs²[t]) .^ 2) for t in 1:T)
        ctrl = 1.25 * sum(sum(u .^ 2) for u in us³)
        goal + stay + ctrl
    end
    Js = Dict{Int,Any}(1 => J₁, 2 => J₂, 3 => J₃)

    function make_constraints(i)
        return function (zᵢ)
            dyn = mapreduce(vcat, 1:T) do t
                (; xs, us) = unflatten_trajectory(zᵢ, state_dim, control_dim)
                xs[t+1] - xs[t] - Δt * us[t]
            end
            (; xs,) = unflatten_trajectory(zᵢ, state_dim, control_dim)
            ic = xs[1] - θs[i]
            vcat(dyn, ic)
        end
    end
    gs = [make_constraints(i) for i in 1:N]

    params = Dict(1 => [-5.0, 1.0], 2 => [-2.0, -2.5], 3 => [2.0, -4.0])
    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, params, N, T, name="PPV (T=$T)")
end

# ═══════════════════════════════════════════════════════
# Problem 3: Nonlinear Lane Change (large) - optional
# ═══════════════════════════════════════════════════════
# Include common utilities needed by lane change
include(joinpath(@__DIR__, "..", "experiments", "common", "dynamics.jl"))
include(joinpath(@__DIR__, "..", "experiments", "common", "collision_avoidance.jl"))
include(joinpath(@__DIR__, "..", "experiments", "common", "trajectory_utils.jl"))

function build_lane_change(; T=10, state_dim=4, control_dim=2, Δt=0.4, R=6.0)
    N = 4
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 4)

    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = fill(primal_dim, N)
    θs = setup_problem_parameter_variables(fill(state_dim, N))

    target_v = 2.0

    function J₁(z₁, z₂, z₃, z₄; θ=nothing)
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, us¹ = xs, us
        (; xs,) = unflatten_trajectory(z₂, state_dim, control_dim); xs² = xs
        (; xs,) = unflatten_trajectory(z₃, state_dim, control_dim); xs³ = xs
        (; xs,) = unflatten_trajectory(z₄, state_dim, control_dim); xs⁴ = xs
        ctrl = 10.0 * sum(sum(u .^ 2) for u in us¹)
        coll = 0.1 * smooth_collision_all(xs¹, xs², xs³, xs⁴)
        vel = 1.0 * sum((x[4] - target_v)^2 for x in xs¹)
        y_dev = 5.0 * sum((x[2] - R)^2 for x in xs¹)
        heading = 1.0 * sum(x[3]^2 for x in xs¹)
        ctrl + coll + y_dev + heading + vel
    end
    function J₂(z₁, z₂, z₃, z₄; θ=nothing)
        (; xs,) = unflatten_trajectory(z₁, state_dim, control_dim); xs¹ = xs
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim); xs², us² = xs, us
        (; xs,) = unflatten_trajectory(z₃, state_dim, control_dim); xs³ = xs
        (; xs,) = unflatten_trajectory(z₄, state_dim, control_dim); xs⁴ = xs
        ctrl = 1.0 * sum(sum(u .^ 2) for u in us²)
        coll = 0.1 * smooth_collision_all(xs¹, xs², xs³, xs⁴)
        vel = 1.0 * sum((x[4] - target_v)^2 for x in xs²)
        y_dev = 5.0 * sum((x[2] - R)^2 for x in xs²)
        heading = 1.0 * sum(x[3]^2 for x in xs²)
        ctrl + coll + y_dev + heading + vel
    end
    function J₃(z₁, z₂, z₃, z₄; θ=nothing)
        (; xs,) = unflatten_trajectory(z₁, state_dim, control_dim); xs¹ = xs
        (; xs,) = unflatten_trajectory(z₂, state_dim, control_dim); xs² = xs
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim); xs³, us³ = xs, us
        (; xs,) = unflatten_trajectory(z₄, state_dim, control_dim); xs⁴ = xs
        tracking = 10.0 * sum((sum(x[1:2] .^ 2) - R^2)^2 for x in xs³[2:div(T,2)])
        ctrl = 1.0 * sum(sum(u .^ 2) for u in us³)
        coll = 0.1 * smooth_collision_all(xs¹, xs², xs³, xs⁴)
        vel = 1.0 * sum((x[4] - target_v)^2 for x in xs³)
        y_dev = 5.0 * sum((x[2] - R)^2 for x in xs³[div(T,2):T])
        heading = 1.0 * sum(x[3]^2 for x in xs³[div(T,2):T])
        tracking + ctrl + coll + y_dev + heading + vel
    end
    function J₄(z₁, z₂, z₃, z₄; θ=nothing)
        (; xs,) = unflatten_trajectory(z₁, state_dim, control_dim); xs¹ = xs
        (; xs,) = unflatten_trajectory(z₂, state_dim, control_dim); xs² = xs
        (; xs,) = unflatten_trajectory(z₃, state_dim, control_dim); xs³ = xs
        (; xs, us) = unflatten_trajectory(z₄, state_dim, control_dim); xs⁴, us⁴ = xs, us
        ctrl = 1.0 * sum(sum(u .^ 2) for u in us⁴)
        coll = 0.1 * smooth_collision_all(xs¹, xs², xs³, xs⁴)
        vel = 1.0 * sum((x[4] - target_v)^2 for x in xs⁴)
        y_dev = sum((x[2] - R)^2 for x in xs⁴)
        heading = 1.0 * sum(x[3]^2 for x in xs⁴)
        ctrl + coll + y_dev + heading + vel
    end
    Js = Dict{Int,Any}(1 => J₁, 2 => J₂, 3 => J₃, 4 => J₄)

    function make_constraints(i)
        return function (zᵢ)
            dyn = mapreduce(vcat, 1:T) do t
                unicycle_dynamics(zᵢ, t; Δt, state_dim, control_dim)
            end
            (; xs,) = unflatten_trajectory(zᵢ, state_dim, control_dim)
            ic = xs[1] - θs[i]
            vcat(dyn, ic)
        end
    end
    gs = [make_constraints(i) for i in 1:N]

    x0 = [
        [-1.5R, R, 0.0, target_v],
        [-2.0R, R, 0.0, target_v],
        [-R, 0.0, π/2, 1.523],
        [-2.5R, R, 0.0, target_v],
    ]
    params = Dict(i => x0[i] for i in 1:N)

    # Build initial guess
    x0_1, u0_1 = make_straight_traj(T, Δt; x0=x0[1])
    x0_2, u0_2 = make_straight_traj(T, Δt; x0=x0[2])
    x0_4, u0_4 = make_straight_traj(T, Δt; x0=x0[4])
    x0_3, u0_3 = make_unicycle_traj(T, Δt; R, split=0.5, x0=x0[3])
    z0_guess = vcat(
        flatten_trajectory(x0_1, u0_1),
        flatten_trajectory(x0_2, u0_2),
        flatten_trajectory(x0_3, u0_3),
        flatten_trajectory(x0_4, u0_4),
    )

    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, params, N, T,
             initial_guess=z0_guess, name="Lane change 4-player (T=$T)")
end

# ═══════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════
function benchmark_problem(prob; num_warmup=3, num_solves=10)
    println("  Building solver...")
    solver = NonlinearSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                             prob.state_dim, prob.control_dim; max_iters=100, tol=1e-8)
    precomputed = solver.precomputed
    n_vars = length(precomputed.all_variables)
    println("  Solver built. $n_vars variables.")

    initial_guess = hasproperty(prob, :initial_guess) ? prob.initial_guess : nothing

    # Warmup both paths
    println("  Warming up...")
    for _ in 1:num_warmup
        run_nonlinear_solver(precomputed, prob.params, prob.G;
            max_iters=100, tol=1e-8, initial_guess=initial_guess)
        run_nonlinear_solver(precomputed, prob.params, prob.G;
            max_iters=100, tol=1e-8, preallocate=true, initial_guess=initial_guess)
    end

    # Benchmark default path
    println("  Benchmarking default ($(num_solves) solves)...")
    to_default = TimerOutput()
    alloc_default = 0
    for _ in 1:num_solves
        alloc_default += @allocated begin
            @timeit to_default "default" begin
                run_nonlinear_solver(precomputed, prob.params, prob.G;
                    max_iters=100, tol=1e-8, initial_guess=initial_guess, to=to_default)
            end
        end
    end

    # Benchmark pre-allocated path
    println("  Benchmarking preallocate ($(num_solves) solves)...")
    to_prealloc = TimerOutput()
    alloc_prealloc = 0
    for _ in 1:num_solves
        alloc_prealloc += @allocated begin
            @timeit to_prealloc "preallocate" begin
                run_nonlinear_solver(precomputed, prob.params, prob.G;
                    max_iters=100, tol=1e-8, preallocate=true, initial_guess=initial_guess,
                    to=to_prealloc)
            end
        end
    end

    # Verify correctness
    result_default = run_nonlinear_solver(precomputed, prob.params, prob.G;
        max_iters=100, tol=1e-8, initial_guess=initial_guess)
    result_prealloc = run_nonlinear_solver(precomputed, prob.params, prob.G;
        max_iters=100, tol=1e-8, preallocate=true, initial_guess=initial_guess)

    sol_diff = norm(result_default.sol - result_prealloc.sol)
    iter_match = result_default.iterations == result_prealloc.iterations

    # Extract timing
    t_default = to_default["default"]
    t_prealloc = to_prealloc["preallocate"]
    ms_default = TimerOutputs.time(t_default) / 1e6 / num_solves
    ms_prealloc = TimerOutputs.time(t_prealloc) / 1e6 / num_solves
    mib_default = alloc_default / 1024^2 / num_solves
    mib_prealloc = alloc_prealloc / 1024^2 / num_solves

    return (;
        ms_default, ms_prealloc,
        mib_default, mib_prealloc,
        speedup = ms_default / ms_prealloc,
        alloc_ratio = mib_default / mib_prealloc,
        sol_diff, iter_match,
        iters = result_default.iterations,
        status = result_default.status,
        to_default, to_prealloc,
    )
end

# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
println("═" ^ 70)
println("  Pre-allocation Benchmark: Default vs preallocate=true")
println("═" ^ 70)

problems = Any[
    build_lq_chain(T=10),
    build_ppv(T=20),
]

if RUN_LANE_CHANGE
    println("\n  Note: Lane change construction can take 5+ minutes and 20+ GB RAM")
    push!(problems, build_lane_change(T=10))
end

results = Dict{String, NamedTuple}()

for prob in problems
    println("\n─── $(prob.name) ───")
    r = benchmark_problem(prob; num_warmup=3, num_solves=10)
    results[prob.name] = r

    println("\n  Results:")
    println("    Default:     $(round(r.ms_default, digits=2)) ms/solve, $(round(r.mib_default, digits=2)) MiB/solve")
    println("    Pre-alloc:   $(round(r.ms_prealloc, digits=2)) ms/solve, $(round(r.mib_prealloc, digits=2)) MiB/solve")
    println("    Speedup:     $(round(r.speedup, digits=3))×")
    println("    Alloc ratio: $(round(r.alloc_ratio, digits=3))×")
    println("    Correctness: Δsol=$(round(r.sol_diff, sigdigits=3)), iters_match=$(r.iter_match)")
    println("    Solver:      $(r.iters) iterations, status=$(r.status)")
end

# Summary table
println("\n" * "═" ^ 70)
println("  SUMMARY TABLE")
println("═" ^ 70)
println()
println("  $(rpad("Problem", 35)) $(lpad("Default ms", 12)) $(lpad("Prealloc ms", 12)) $(lpad("Speedup", 8)) $(lpad("Alloc Δ", 8)) $(lpad("Δsol", 10))")
println("  " * "─" ^ 85)

for prob in problems
    r = results[prob.name]
    marker = r.speedup > 1.05 ? " ✓" : (r.speedup < 0.95 ? " ✗" : " ~")
    println("  $(rpad(prob.name, 35)) $(lpad(round(r.ms_default, digits=2), 12)) $(lpad(round(r.ms_prealloc, digits=2), 12)) $(lpad(string(round(r.speedup, digits=2), "×"), 8)) $(lpad(string(round(r.alloc_ratio, digits=2), "×"), 8)) $(lpad(round(r.sol_diff, sigdigits=3), 10))$marker")
end

# Detailed TimerOutputs for each problem
println("\n" * "═" ^ 70)
println("  DETAILED TIMER OUTPUTS")
println("═" ^ 70)

for prob in problems
    r = results[prob.name]
    println("\n  ┌─ $(prob.name) (default) ─────────")
    show(r.to_default)
    println()
    println("\n  ┌─ $(prob.name) (preallocate) ─────────")
    show(r.to_prealloc)
    println()
end

println("\n═" ^ 70)
println("  CONCLUSION")
println("═" ^ 70)

any_speedup = all(r -> r.speedup > 1.05, values(results))
any_slowdown = any(r -> r.speedup < 0.90, values(results))
if any_slowdown
    println("\n  Pre-allocation causes SLOWDOWN on at least one problem.")
    println("  Persistent Dict caches increase GC pressure during intensive iteration loops.")
    println("  The dominant allocations come from SymbolicTracingUtils compiled functions")
    println("  (M_fns, N_fns) and the `\\` operator, which cannot be pre-allocated.")
    println("  Recommend NOT landing the pre-allocation flag.")
elseif any_speedup
    println("\n  Pre-allocation shows consistent speedup across all problems.")
    println("  Consider landing the pre-allocation flag.")
else
    println("\n  Pre-allocation shows NO meaningful speedup on tested problems.")
    println("  The dominant allocations come from SymbolicTracingUtils compiled functions")
    println("  (M_fns, N_fns) and the `\\` operator, which cannot be pre-allocated.")
    println("  Recommend NOT landing the pre-allocation flag.")
end

println("\n═" ^ 70)
