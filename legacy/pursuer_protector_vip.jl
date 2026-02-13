using LinearAlgebra: norm
using TrajectoryGamesBase: unflatten_trajectory

include("automatic_solver.jl") # brings in get_three_player_openloop_lq_problem, solve_nonlq_game_example, plot_player_trajectories, evaluate_kkt_residuals

"""
Run the pursuer–protector–VIP pursuit/defense scenario.

Arguments (keyword):
- `T`: horizon length (steps).
- `Δt`: step duration.
- `x0`: initial states per player (Vector of 2-vectors) ordered [pursuer, protector, VIP].
- `x_goal`: desired goal for the VIP.
- `run_lq`: if true, also solve the LQ reference version (same dynamics, quadraticized objective).
- `verbose`, `show_timing_info`: pass through to solver for debugging.
"""
function run_pursuer_protector_vip(; T=20, Δt=0.1,
	x0 = [
		[-5.0; 1.0],   # pursuer
		[-2.0; -2.5],  # protector
		[ 2.0; -4.0],  # VIP
	],
	x_goal = [0.0; 0.0],
	run_lq=false, verbose=false, show_timing_info=false,
)
	N, G, H, problem_dims, Js_base, gs, θs, backend = get_three_player_openloop_lq_problem(T, Δt; verbose=false)

	state_dimension = problem_dims.state_dimension
	control_dimension = problem_dims.control_dimension
	primal_dimension_per_player = problem_dims.primal_dimension_per_player

	# Custom objectives for pursuit/defense
	Js = Dict{Int, Any}()

	# Pursuer: chase VIP, lightly repulse protector, penalize control effort.
	Js[1] = (z₁, z₂, z₃, θi) -> begin
		(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension); xs¹, us¹ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension); xs², _ = xs, us
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension); xs³, _ = xs, us
		2sum(sum((xs³[t] - xs¹[t]).^2 for t in 1:T)) -
		sum(sum((xs²[t] - xs¹[t]).^2 for t in 1:T)) +
		1.25*sum(sum(u .^ 2) for u in us¹)
	end

	# Protector: stay with VIP, pull VIP away from pursuer, moderate control effort.
	Js[2] = (z₁, z₂, z₃, θi) -> begin
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension); xs², us² = xs, us
		(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension); xs¹, _ = xs, us
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension); xs³, _ = xs, us
		0.5sum(sum((xs³[t] - xs²[t]).^2 for t in 1:T)) -
		sum(sum((xs³[t] - xs¹[t]).^2 for t in 1:T)) +
		0.25*sum(sum(u .^ 2) for u in us²)
	end

	# VIP: reach goal, stay close to protector, penalize control.
	Js[3] = (z₁, z₂, z₃, θi) -> begin
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension); xs³, us³ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension); xs², _ = xs, us
		out = 10*sum((xs³[end] .- x_goal) .^ 2) + 1.25*sum(sum(u .^ 2) for u in us³)
		out += sum(sum((xs³[t] - xs²[t]).^2 for t in 1:T)) # stay close to protector
		out
	end

	@info "Pursuer–Protector–VIP scenario" N H T Δt

	parameter_values = x0
	if run_lq
		z_sol_nonlq, status_nonlq, z_sol_lq, status_lq, info_nonlq, info_lq, all_variables, vars, all_augmented_vars =
			compare_lq_and_nonlq_solver(H, G, primal_dimension_per_player, Js, gs, θs, parameter_values, backend; verbose)
	else
		z_sol_nonlq, status_nonlq, info_nonlq, all_variables, vars, all_augmented_vars =
			solve_nonlq_game_example(H, G, primal_dimension_per_player, Js, gs, θs, parameter_values; verbose)
		@info "Non-LQ solver status after $(info_nonlq.num_iterations) iterations: $(status_nonlq)"
		show_timing_info && show(info_nonlq.to)
		z_sol_lq = nothing; status_lq = nothing; info_lq = nothing
	end

	z_sol = z_sol_nonlq
	(; πs, zs, λs, μs, θs) = vars
	(; out_all_augment_variables, out_all_augmented_z_est) = all_augmented_vars

	# Split per-player solutions
	z_sols = Vector{Vector{Float64}}(undef, N)
	offs = 1
	for i in 1:N
		li = length(zs[i])
		z_sols[i] = @view z_sol[offs:offs+li-1]
		offs += li
	end

	# Residuals
	if run_lq
		πs_eval_lq = strip_policy_constraints(info_lq.πs, G, zs, gs)
		evaluate_kkt_residuals(πs_eval_lq, all_variables, z_sol_lq, θs, parameter_values; verbose = true)
	end
	πs_eval = strip_policy_constraints(πs, G, zs, gs)
	evaluate_kkt_residuals(πs_eval, out_all_augment_variables, out_all_augmented_z_est, θs, parameter_values; verbose = true)

	# Plot
	plot_player_trajectories(z_sols, T, Δt, problem_dims)

	return (; z_sol, z_sols, status_nonlq, status_lq, info_nonlq, info_lq)
end

if abspath(PROGRAM_FILE) == @__FILE__
	run_pursuer_protector_vip()
end
