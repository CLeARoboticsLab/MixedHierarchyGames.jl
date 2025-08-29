using ParametricMCPs: ParametricMCPs
using PATHSolver: PATHSolver
using LinearAlgebra: I, norm
using Symbolics
using SymbolicTracingUtils
using TrajectoryGamesBase: unflatten_trajectory
using InvertedIndices
using Statistics
using Plots
using Random

# Set seed for reproducibility
Random.seed!(42)

# ------------------------------------------------------------
# Problem setup
# ------------------------------------------------------------
num_players                 = 3
T                           = 3
state_dimension             = 2
control_dimension           = 2
x_dim                       = state_dimension * (T + 1)
u_dim                       = control_dimension * (T + 1)
primal_dimension_per_player = x_dim + u_dim

# simple single-integrator dynamics step: x_{t+1} = x_t + Δt * u_t
Δt = 0.5

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
flatten(vs) = collect(Iterators.flatten(vs))

"Unpack a trajectory vector z into (xs, us) arrays (length T+1 each)."
unpack(z) = unflatten_trajectory(z, state_dimension, control_dimension)

"Vectorize all players' states/controls for quadratic costs."
function all_xu(z₁, z₂, z₃)
    xs¹, us¹ = unpack(z₁)
    xs², us² = unpack(z₂)
    xs³, us³ = unpack(z₃)
    x = vcat(xs¹..., xs²..., xs³...)
    u = vcat(us¹..., us²..., us³...)
    return x, u
end

"Per-player discrete dynamics residuals stacked over t=1:T."
g(z) = mapreduce(vcat, 1:T) do t
    xs, us = unpack(z)
    xs[t+1] .- xs[t] .- Δt .* us[t]
end

"Build one player's quadratic cost from (Q,R)."
function J_check(z₁, z₂, z₃, Q, R)
    x, u = all_xu(z₁, z₂, z₃)
    0.5 * x' * (Q' * Q) * x + 0.5 * u' * (R' * R) * u
end

"Make a contiguous split (v -> v[1:n1], v[n1+1:n1+n2], v[...])."
split3(v, n1, n2, n3) = (v[1:n1], v[n1+1:n1+n2], v[n1+n2+1:n1+n2+n3])

"Thin wrapper around PATH solve."
function solve_path(F, vars, θ; guess=nothing)
    z̲ = fill(-Inf, length(F))
    z̅ = fill( Inf, length(F))
    parametric = ParametricMCPs.ParametricMCP(F, vars, [θ], z̲, z̅; compute_sensitivities=false)
    z0 = isnothing(guess) ? zeros(length(vars)) : guess
    ParametricMCPs.solve(
        parametric, [1e-5];
        initial_guess = z0,
        verbose=false,
        cumulative_iteration_limit=100_000,
        proximal_perturbation=1e-2,
        use_basics=true,
        use_start=true,
    )
end

"Pretty print a player's (x,u) and objective."
function print_solution(label, z, J)
    xs, us = unpack(z)
    println("$label (x,u): ($xs, $us)")
    println("$label Objective: $(J)")
end

# ------------------------------------------------------------
# Function to run single instance
# ------------------------------------------------------------
function run_single_instance(Q¹, R¹, Q², R², Q³, R³, z₁, z₂, z₃, λ₁, λ₂, λ₃, 
                            λₙ₁, λₙ₂, λₙ₃, F_stack, vars_stack, F_nash, vars_nash, θ)
    
    # Define cost functions for this instance
    J¹(z1,z2,z3) = J_check(z1, z2, z3, Q¹, R¹)
    J²(z1,z2,z3) = J_check(z1, z2, z3, Q², R²)
    J³(z1,z2,z3) = J_check(z1, z2, z3, Q³, R³)
    
    # Solve Stackelberg
    z_all_s, status_s, info_s = solve_path(F_stack, vars_stack, θ)
    
    # Solve Nash
    z_all_n, status_n, info_n = solve_path(F_nash, vars_nash, θ)
    
    # Extract solutions
    z₁_s, z₂_s, z₃_s = split3(z_all_s, length(z₁), length(z₂), length(z₃))
    z₁_n, z₂_n, z₃_n = split3(z_all_n, length(z₁), length(z₂), length(z₃))
    
    # Compute objectives
    obj_s = [J¹(z₁_s, z₂_s, z₃_s), J²(z₁_s, z₂_s, z₃_s), J³(z₁_s, z₂_s, z₃_s)]
    obj_n = [J¹(z₁_n, z₂_n, z₃_n), J²(z₁_n, z₂_n, z₃_n), J³(z₁_n, z₂_n, z₃_n)]
    
    # Compute differences
    traj_diff = [norm(z₁_s .- z₁_n, 2), norm(z₂_s .- z₂_n, 2), norm(z₃_s .- z₃_n, 2)]
    obj_diff = abs.(obj_s .- obj_n)
    total_traj_diff = sum(traj_diff)
    
    return (
        converged_stack = (status_s == PATHSolver.MCP_Solved),
        converged_nash = (status_n == PATHSolver.MCP_Solved),
        traj_diff = traj_diff,
        total_traj_diff = total_traj_diff,
        obj_stack = obj_s,
        obj_nash = obj_n,
        obj_diff = obj_diff,
        z_stack = [z₁_s, z₂_s, z₃_s],
        z_nash = [z₁_n, z₂_n, z₃_n]
    )
end

# ------------------------------------------------------------
# Setup symbolic problem once
# ------------------------------------------------------------
backend = SymbolicTracingUtils.SymbolicsBackend()
z₁ = SymbolicTracingUtils.make_variables(backend, Symbol("z̃1"), primal_dimension_per_player)
z₂ = SymbolicTracingUtils.make_variables(backend, Symbol("z̃2"), primal_dimension_per_player)
z₃ = SymbolicTracingUtils.make_variables(backend, Symbol("z̃3"), primal_dimension_per_player)
symbolic_type = eltype(z₁)
θ = only(SymbolicTracingUtils.make_variables(backend, :θ, 1))

# Initial conditions (as equality constraints)
xs¹, _ = unpack(z₁); xs², _ = unpack(z₂); xs³, _ = unpack(z₃)
ic₁ = xs¹[1] .- [0.0; 2.0]
ic₂ = xs²[1] .- [2.0; 4.0]
ic₃ = xs³[1] .- [6.0; 8.0]

# Dimensions for cost matrices
total_x_dim = num_players * state_dimension * (T + 1)
total_u_dim = num_players * control_dimension * (T + 1)

# Create a placeholder instance to build symbolic expressions
Q¹_sym, R¹_sym = rand(total_x_dim, total_x_dim), rand(total_u_dim, total_u_dim)
Q²_sym, R²_sym = rand(total_x_dim, total_x_dim), rand(total_u_dim, total_u_dim)
Q³_sym, R³_sym = rand(total_x_dim, total_x_dim), rand(total_u_dim, total_u_dim)

J¹_sym(z1,z2,z3) = J_check(z1, z2, z3, Q¹_sym, R¹_sym)
J²_sym(z1,z2,z3) = J_check(z1, z2, z3, Q²_sym, R²_sym)
J³_sym(z1,z2,z3) = J_check(z1, z2, z3, Q³_sym, R³_sym)

# Build Stackelberg symbolic system
λ₃ = SymbolicTracingUtils.make_variables(backend, Symbol("λ_3"), length(g(z₃)) + length(ic₃))
L₃ = J³_sym(z₁, z₂, z₃) - λ₃' * [g(z₃); ic₃]
π₃ = vcat( Symbolics.gradient(L₃, z₃), g(z₃), ic₃ )

p₂_eq_dim   = length(g(z₂)) + length(ic₂)
p₂_stat_dim = length(z₃)
λ₂₁ = SymbolicTracingUtils.make_variables(backend, Symbol("λ_2_1"), p₂_eq_dim)
λ₂₂ = SymbolicTracingUtils.make_variables(backend, Symbol("λ_2_2"), p₂_stat_dim)
λ₂  = vcat(λ₂₁, λ₂₂)

M₂ = Symbolics.jacobian(π₃, [z₃; λ₃])
N₂ = Symbolics.jacobian(π₃, [z₁; z₂])

S₃ = hcat(I(length(z₃)), zeros(length(z₃), length(λ₃)))
ϕ³ = - S₃ * (M₂ \ N₂ * [z₁; z₂])

L² = J²_sym(z₁, z₂, z₃) - λ₂₁' * [g(z₂); ic₂] - λ₂₂' * (z₃ .- ϕ³)
π₂ = vcat( Symbolics.gradient(L², z₂),
           Symbolics.gradient(L², z₃),
           g(z₂), ic₂ )

p₁_eq_dim   = length(g(z₁)) + length(ic₁)
p₁_stat₂    = length(z₂)
p₁_stat₃    = length(z₃)
λ₁₁ = SymbolicTracingUtils.make_variables(backend, Symbol("λ_1_1"), p₁_eq_dim)
λ₁₂ = SymbolicTracingUtils.make_variables(backend, Symbol("λ_1_2"), p₁_stat₂)
λ₁₃ = SymbolicTracingUtils.make_variables(backend, Symbol("λ_1_3"), p₁_stat₃)
λ₁  = vcat(λ₁₁, λ₁₂, λ₁₃)

M₁ = Symbolics.jacobian(π₂, [z₂; z₃; λ₂; λ₃])
N₁ = Symbolics.jacobian(π₂, z₁)

S₂ = hcat(I(length(z₂)), zeros(length(z₂), length(z₃) + length(λ₂) + length(λ₃)))
ϕ² = - S₂ * (M₁ \ N₁ * z₁)

L¹ = J¹_sym(z₁, z₂, z₃) - λ₁₁' * [g(z₁); ic₁] - λ₁₂' * (z₂ .- ϕ²) - λ₁₃' * (z₃ .- ϕ³)
π₁ = vcat( Symbolics.gradient(L¹, z₁),
           Symbolics.gradient(L¹, z₂),
           Symbolics.gradient(L¹, z₃),
           g(z₁), ic₁ )

F_stack = Vector{symbolic_type}([π₁; π₂; π₃])
vars_stack = vcat(z₁, z₂, z₃, λ₁, λ₂, λ₃)

# Build Nash symbolic system
λₙ₁ = SymbolicTracingUtils.make_variables(backend, Symbol("λₙ_1"), length(g(z₁)) + length(ic₁))
λₙ₂ = SymbolicTracingUtils.make_variables(backend, Symbol("λₙ_2"), length(g(z₂)) + length(ic₂))
λₙ₃ = SymbolicTracingUtils.make_variables(backend, Symbol("λₙ_3"), length(g(z₃)) + length(ic₃))

Lₙ₁ = J¹_sym(z₁, z₂, z₃) - λₙ₁' * [g(z₁); ic₁]
Lₙ₂ = J²_sym(z₁, z₂, z₃) - λₙ₂' * [g(z₂); ic₂]
Lₙ₃ = J³_sym(z₁, z₂, z₃) - λₙ₃' * [g(z₃); ic₃]

F_nash = Vector{symbolic_type}([
    Symbolics.gradient(Lₙ₁, z₁); g(z₁); ic₁;
    Symbolics.gradient(Lₙ₂, z₂); g(z₂); ic₂;
    Symbolics.gradient(Lₙ₃, z₃); g(z₃); ic₃;
])
vars_nash = vcat(z₁, z₂, z₃, λₙ₁, λₙ₂, λₙ₃)

# ------------------------------------------------------------
# Monte Carlo Study
# ------------------------------------------------------------
n_trials = 100
results = []

println("Starting Monte Carlo study with $n_trials trials...")
println("="^60)

for trial in 1:n_trials
    # Generate random cost matrices
    local Q¹ = randn(total_x_dim, total_x_dim) 
    local R¹ = randn(total_u_dim, total_u_dim) 
    local Q² = randn(total_x_dim, total_x_dim) 
    local R² = randn(total_u_dim, total_u_dim) 
    local Q³ = randn(total_x_dim, total_x_dim) 
    local R³ = randn(total_u_dim, total_u_dim) 
    
    # Update the symbolic expressions with new matrices
    Q¹_sym .= Q¹; R¹_sym .= R¹
    Q²_sym .= Q²; R²_sym .= R²
    Q³_sym .= Q³; R³_sym .= R³
    
    result = run_single_instance(Q¹, R¹, Q², R², Q³, R³, z₁, z₂, z₃, λ₁, λ₂, λ₃,
                                 λₙ₁, λₙ₂, λₙ₃, F_stack, vars_stack, F_nash, vars_nash, θ)
    push!(results, result)
    
    if trial % 10 == 0
        println("Completed trial $trial/$n_trials")
    end
end

# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------
println("\n" * "="^60)
println("MONTE CARLO RESULTS SUMMARY")
println("="^60)

# Convergence statistics
stack_converged = sum(r.converged_stack for r in results)
nash_converged = sum(r.converged_nash for r in results)
both_converged = sum(r.converged_stack && r.converged_nash for r in results)

println("\nConvergence Statistics:")
println("  Stackelberg converged: $stack_converged/$n_trials ($(100*stack_converged/n_trials)%)")
println("  Nash converged: $nash_converged/$n_trials ($(100*nash_converged/n_trials)%)")
println("  Both converged: $both_converged/$n_trials ($(100*both_converged/n_trials)%)")

# Filter to only converged cases
converged_results = filter(r -> r.converged_stack && r.converged_nash, results)
n_converged = length(converged_results)

if n_converged > 0
    # Trajectory differences
    traj_diffs = [r.total_traj_diff for r in converged_results]
    println("\nTrajectory Differences (L2 norm):")
    println("  Mean: $(mean(traj_diffs))")
    println("  Std:  $(std(traj_diffs))")
    println("  Min:  $(minimum(traj_diffs))")
    println("  Max:  $(maximum(traj_diffs))")
    
    # Per-player trajectory differences
    player_traj_diffs = hcat([r.traj_diff for r in converged_results]...)'
    println("\nPer-Player Trajectory Differences:")
    for p in 1:3
        println("  Player $p - Mean: $(mean(player_traj_diffs[:,p])), Std: $(std(player_traj_diffs[:,p]))")
    end
    
    # Objective differences
    obj_diffs = hcat([r.obj_diff for r in converged_results]...)'
    println("\nObjective Value Differences:")
    for p in 1:3
        println("  Player $p - Mean: $(mean(obj_diffs[:,p])), Std: $(std(obj_diffs[:,p]))")
    end
    
    # Count significant differences
    threshold = 1e-3
    significant_diffs = sum(traj_diffs .> threshold)
    println("\nSignificant Differences (trajectory diff > $threshold):")
    println("  $significant_diffs/$n_converged ($(100*significant_diffs/n_converged)%)")
    
    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------
    println("\nGenerating visualizations...")
    
    # Create subplots
    p1 = histogram(traj_diffs, 
                   xlabel="Total Trajectory Difference (L2)", 
                   ylabel="Frequency",
                   title="Distribution of Solution Differences",
                   legend=false,
                   bins=20)
    
    p2 = boxplot(["Player 1" "Player 2" "Player 3"], player_traj_diffs',
                 ylabel="Trajectory Difference (L2)",
                 title="Per-Player Trajectory Differences",
                 legend=false)
    
    p3 = boxplot(["Player 1" "Player 2" "Player 3"], obj_diffs',
                 ylabel="Objective Difference",
                 title="Per-Player Objective Differences",
                 legend=false)
    
    # Scatter plot of objective values
    stack_objs = hcat([r.obj_stack for r in converged_results]...)'
    nash_objs = hcat([r.obj_nash for r in converged_results]...)'
    
    p4 = scatter(nash_objs[:,1], stack_objs[:,1], 
                 xlabel="Nash Objective", ylabel="Stackelberg Objective",
                 title="Player 1 (Leader) Objectives",
                 label="Player 1", markersize=3, alpha=0.6)
    plot!(p4, nash_objs[:,1], nash_objs[:,1], label="y=x", linestyle=:dash, color=:black)
    
    p5 = scatter(nash_objs[:,2], stack_objs[:,2],
                 xlabel="Nash Objective", ylabel="Stackelberg Objective", 
                 title="Player 2 (Middle) Objectives",
                 label="Player 2", markersize=3, alpha=0.6, color=:orange)
    plot!(p5, nash_objs[:,2], nash_objs[:,2], label="y=x", linestyle=:dash, color=:black)
    
    p6 = scatter(nash_objs[:,3], stack_objs[:,3],
                 xlabel="Nash Objective", ylabel="Stackelberg Objective",
                 title="Player 3 (Follower) Objectives", 
                 label="Player 3", markersize=3, alpha=0.6, color=:green)
    plot!(p6, nash_objs[:,3], nash_objs[:,3], label="y=x", linestyle=:dash, color=:black)
    
    # Combine plots
    plot_combined = plot(p1, p2, p3, p4, p5, p6, 
                        layout=(3,2), size=(1000, 900),
                        plot_title="Monte Carlo Study: Stackelberg vs Nash Equilibrium")
    
    display(plot_combined)
    
    # Additional analysis plot: CDF of differences
    sorted_diffs = sort(traj_diffs)
    cdf_vals = (1:n_converged) / n_converged
    
    p_cdf = plot(sorted_diffs, cdf_vals,
                 xlabel="Total Trajectory Difference (L2)",
                 ylabel="Cumulative Probability",
                 title="CDF of Solution Differences",
                 legend=false,
                 linewidth=2)
    vline!([threshold], linestyle=:dash, color=:red, label="Threshold")
    
    display(p_cdf)
    
    # Show example trajectories from most different case
    max_diff_idx = argmax(traj_diffs)
    max_diff_result = converged_results[max_diff_idx]
    
    println("\n" * "="^60)
    println("MOST DIFFERENT CASE (Trial with max trajectory difference)")
    println("="^60)
    println("Total trajectory difference: $(max_diff_result.total_traj_diff)")
    
    # Plot trajectories for the most different case
    plot_trajectories = plot(layout=(3,2), size=(1000, 600),
                            plot_title="Trajectories: Most Different Case")
    
    for (p_idx, p_name) in enumerate(["Player 1 (Leader)", "Player 2 (Middle)", "Player 3 (Follower)"])
        xs_stack, us_stack = unpack(max_diff_result.z_stack[p_idx])
        xs_nash, us_nash = unpack(max_diff_result.z_nash[p_idx])
        
        # State trajectories
        subplot_idx = 2*(p_idx-1) + 1
        plot!(plot_trajectories, subplot=subplot_idx,
              title="$p_name - States",
              xlabel="Time", ylabel="State")
        for i in 1:length(xs_stack)
            plot!(plot_trajectories, subplot=subplot_idx,
                  [xs_stack[i][1]], [xs_stack[i][2]], 
                  seriestype=:scatter, label=(i==1 ? "Stackelberg" : ""),
                  color=:blue, markersize=4)
            plot!(plot_trajectories, subplot=subplot_idx,
                  [xs_nash[i][1]], [xs_nash[i][2]], 
                  seriestype=:scatter, label=(i==1 ? "Nash" : ""),
                  color=:red, markersize=4)
        end
        
        # Control trajectories
        subplot_idx = 2*p_idx
        plot!(plot_trajectories, subplot=subplot_idx,
              title="$p_name - Controls",
              xlabel="Time", ylabel="Control")
        t_vals = 0:T
        us_stack_1 = [u[1] for u in us_stack]
        us_stack_2 = [u[2] for u in us_stack]
        us_nash_1 = [u[1] for u in us_nash]
        us_nash_2 = [u[2] for u in us_nash]
        
        plot!(plot_trajectories, subplot=subplot_idx,
              t_vals, us_stack_1, label="Stack u₁", linewidth=2, color=:blue)
        plot!(plot_trajectories, subplot=subplot_idx,
              t_vals, us_stack_2, label="Stack u₂", linewidth=2, color=:lightblue)
        plot!(plot_trajectories, subplot=subplot_idx,
              t_vals, us_nash_1, label="Nash u₁", linewidth=2, color=:red, linestyle=:dash)
        plot!(plot_trajectories, subplot=subplot_idx,
              t_vals, us_nash_2, label="Nash u₂", linewidth=2, color=:pink, linestyle=:dash)
    end
    
    display(plot_trajectories)
    
else
    println("\nNo trials with both solutions converged!")
end

println("\nMonte Carlo study complete!")