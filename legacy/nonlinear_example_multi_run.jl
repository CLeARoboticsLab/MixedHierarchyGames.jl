# Runs the nonlinear lane change example multiple times with perturbed initial states
# and plots convergence statistics on a log y-axis.

using Random
using Statistics
using Plots

include(joinpath(@__DIR__, "test_automatic_solver.jl"))

######### INPUT: Initial conditions (from nonlinear_lane_change.jl) ##########################
R = 6.0  # turning radius
x_goal = [1.5R; R; 0.0; 0.0]  # target position
x0_base = [
	[-1.5R; R; 0.0; 2.0], # P1 (LEADER)
	[-2.0R; R; 0.0; 2.0], # P2 (FOLLOWER)
	[-R; 0.0; pi/2; 1.523], # P3 (LANE MERGER)
    [-2.5R; R; 0.0; 2.0], # P4 (EXTRA PLAYER BEHIND P2)
]

T = 14
Δt = 0.4

function make_unicycle_traj(
    T::Integer,
    Δt::Real;
    R::Real = 6.0,
    split::Real = 0.5,
    x0::AbstractVector{<:Real} = [-R; 0.0; π/2; 1.523],
)
    @assert Δt > 0 "Δt must be positive"
    @assert T ≥ 2 "Horizon T must be at least 2"
    @assert length(x0) == 4 "x0 must be [x, y, ψ, v]"

    T1 = Int(ceil(split * T))
    T1 = clamp(T1, 1, T-1)
    T2 = T - T1

    Δθ = (π/2) / T1
    xs_pos = Vector{Float64}()
    ys_pos = Vector{Float64}()
    for k in 0:T1
        θ = π - k * Δθ
        push!(xs_pos, R * cos(θ))
        push!(ys_pos, R * sin(θ))
    end
    for s in 1:T2
        x = 9.0 * s / T2
        y = R
        push!(xs_pos, x)
        push!(ys_pos, y)
    end
    @assert length(xs_pos) == T + 1

    ψ = Vector{Float64}(undef, T)
    v = Vector{Float64}(undef, T)
    for k in 0:T-1
        dx = xs_pos[k+2] - xs_pos[k+1]
        dy = ys_pos[k+2] - ys_pos[k+1]
        ψ[k+1] = atan(dy, dx)
        v[k+1] = hypot(dx, dy) / Δt
    end

    ψT = ψ[end]
    vT = v[end]

    angle_diff = (a, b) -> atan(sin(a - b), cos(a - b))

    ω = Vector{Float64}(undef, T)
    a = Vector{Float64}(undef, T)
    for t in 1:T
        ψ_prev = (t == 1) ? x0[3] : ψ[t-1]
        v_prev = (t == 1) ? x0[4] : v[t-1]

        ψ_curr = (t == T) ? ψT : ψ[t]
        v_curr = (t == T) ? vT : v[t]

        ω[t] = angle_diff(ψ_curr, ψ_prev) / Δt
        a[t] = (v_curr - v_prev) / Δt
    end

    xs = Vector{Vector{Float64}}(undef, T + 1)
    us = Vector{Vector{Float64}}(undef, T)

    xs[1] = collect(x0)

    for k in 2:T
        xs[k] = [xs_pos[k], ys_pos[k], ψ[k-1], v[k-1]]
    end

    xs[T+1] = [xs_pos[T+1], ys_pos[T+1], ψT, vT]

    for t in 1:T
        us[t] = [a[t], ω[t]]
    end

    us = vcat(us, [[0.0, 0.0]])

    return xs, us
end

"""
	make_straight_traj(T, Δt; x0)

Generate a dynamically feasible straight-line (horizontal) trajectory for the unicycle model.
Positions advance in `x` with constant heading and speed taken from `x0`; controls remain zero.
"""
function make_straight_traj(
	T::Integer,
	Δt::Real;
	x0::AbstractVector{<:Real} = [0.0; 0.0; 0.0; 1.0],
)
	@assert Δt > 0 "Δt must be positive"
	@assert T ≥ 1 "Horizon T must be at least 1"
	@assert length(x0) == 4 "x0 must be [x, y, ψ, v]"

	ψ0 = Float64(x0[3])
	v0 = Float64(x0[4])

	xs = Vector{Vector{Float64}}(undef, T + 1)
	us = Vector{Vector{Float64}}(undef, T)

	xs[1] = collect(x0)

	for k in 2:(T + 1)
		Δ = (k - 1) * Δt
		x = x0[1] + v0 * cos(ψ0) * Δ
		y = x0[2] + v0 * sin(ψ0) * Δ
		xs[k] = [x, y, ψ0, v0]
	end

	for t in 1:T
		us[t] = [0.0, 0.0]
	end
	us = vcat(us, [[0.0, 0.0]])

	return xs, us
end

# Initial guess for all players (based on the base initial conditions)
x0_1, u0_1 = make_straight_traj(T, Δt; x0 = x0_base[1])
z0_guess_1 = vcat([vcat(x0_1[t], u0_1[t]) for t in 1:T+1]...)
x0_2, u0_2 = make_straight_traj(T, Δt; x0 = x0_base[2])
z0_guess_2 = vcat([vcat(x0_2[t], u0_2[t]) for t in 1:T+1]...)
x0_3, u0_3 = make_unicycle_traj(T, Δt; R, split = 0.5, x0 = x0_base[3])
z0_guess_3 = vcat([vcat(x0_3[t], u0_3[t]) for t in 1:T+1]...)
x0_4, u0_4 = make_straight_traj(T, Δt; x0 = x0_base[4])
z0_guess_4 = vcat([vcat(x0_4[t], u0_4[t]) for t in 1:T+1]...)
z0_guess = vcat(z0_guess_1, z0_guess_2, z0_guess_3, z0_guess_4)
#############################################################################################

function build_nonlinear_lane_change_problem(x_goal, R, T, Δt; backend = SymbolicTracingUtils.SymbolicsBackend())
	N = 4

	G = SimpleDiGraph(N)
	add_edge!(G, 1, 2) # P1 -> P2
	add_edge!(G, 2, 4) # P2 -> P4

	H = 1

	state_dimension = 4
	control_dimension = 2

	x_dim = state_dimension * (T+1)
	u_dim = control_dimension * (T+1)
	aggre_state_dimension = x_dim * N
	aggre_control_dimension = u_dim * N
	total_dimension = aggre_state_dimension + aggre_control_dimension
	primal_dimension_per_player = x_dim + u_dim

	println("Number of players: $N")
	println("Number of Stages: $H (OL = 1; FB > 1)")
	println("Time Horizon (# steps): $T")
	println("Step period: Δt = $(Δt)s")
	println("Dimension per player: $(primal_dimension_per_player)")
	println("Total primal dimension: $total_dimension")

	function J₁(z₁, z₂, z₃, z₄, θ)
		(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
		xs¹, us¹ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
		xs², us² = xs, us
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
		xs³, us³ = xs, us
		(; xs, us) = unflatten_trajectory(z₄, state_dimension, control_dimension)
		xs⁴, us⁴ = xs, us

		control = 10sum(sum(u .^ 2) for u in us¹)
		collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
		velocity = sum((x¹[4] - 2.0)^2 for x¹ in xs¹)
		y_deviation = sum((x¹[2]-R)^2 for x¹ in xs¹)
		zero_heading = sum((x¹[3])^2 for x¹ in xs¹)

		control + collision + 5y_deviation + zero_heading + velocity
	end

	function J₂(z₁, z₂, z₃, z₄, θ)
		(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
		xs¹, us¹ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
		xs², us² = xs, us
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
		xs³, us³ = xs, us
		(; xs, us) = unflatten_trajectory(z₄, state_dimension, control_dimension)
		xs⁴, us⁴ = xs, us

		control = sum(sum(u .^ 2) for u in us²)
		collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
		velocity = sum((x²[4] - 2.0)^2 for x² in xs²)
		y_deviation = sum((x²[2]-R)^2 for x² in xs²)
		zero_heading = sum((x²[3])^2 for x² in xs²)

		control + collision + 5y_deviation + zero_heading + velocity
	end

	function J₃(z₁, z₂, z₃, z₄, θ)
		(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
		xs¹, us¹ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
		xs², us² = xs, us
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
		xs³, us³ = xs, us
		(; xs, us) = unflatten_trajectory(z₄, state_dimension, control_dimension)
		xs⁴, us⁴ = xs, us

		tracking = 10sum((sum(x³[1:2] .^ 2) - R^2)^2 for x³ in xs³[2:div(T, 2)])
		control = sum(sum(u³ .^ 2) for u³ in us³)
		collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
		velocity = sum((x³[4] - 2.0)^2 for x³ in xs³)
		y_deviation = sum((x³[2]-R)^2 for x³ in xs³[div(T, 2):T])
		zero_heading = sum((x³[3])^2 for x³ in xs³[div(T, 2):T])

		tracking + control + collision + 5y_deviation + zero_heading + velocity
	end

	function J₄(z₁, z₂, z₃, z₄, θ)
		(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
		xs¹, us¹ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
		xs², us² = xs, us
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
		xs³, us³ = xs, us
		(; xs, us) = unflatten_trajectory(z₄, state_dimension, control_dimension)
		xs⁴, us⁴ = xs, us

		control = sum(sum(u .^ 2) for u in us⁴)
		collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
		velocity = sum((x⁴[4] - 2.0)^2 for x⁴ in xs⁴)
		y_deviation = sum((x⁴[2]-R)^2 for x⁴ in xs⁴)
		zero_heading = sum((x⁴[3])^2 for x⁴ in xs⁴)

		control + collision + y_deviation + zero_heading + velocity
	end

	Js = Dict{Int, Any}(
		1 => J₁,
		2 => J₂,
		3 => J₃,
		4 => J₄,
	)

	num_params_per_player = fill(state_dimension, N)
	θs = setup_problem_parameter_variables(backend, num_params_per_player; verbose = false)

	function unicycle_dynamics(z, t; Δt = Δt)
		(; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
		x_t = xs[t]
		u_t = us[t]
		x_tp1 = xs[t+1]

		x, y, ψ, v = x_t
		a, ω = u_t

		xdot = v * cos(ψ)
		ydot = v * sin(ψ)
		psidot = ω
		vdot = a

		x_pred = x_t .+ Δt .* [xdot, ydot, psidot, vdot]

		return x_tp1 - x_pred
	end

	make_ic_constraint(i) = function (zᵢ)
		(; xs, us) = unflatten_trajectory(zᵢ, state_dimension, control_dimension)
		x1 = xs[1]
		return x1 - θs[i]
	end

	dynamics_constraint(zᵢ) =
		mapreduce(vcat, 1:T) do t
			unicycle_dynamics(zᵢ, t)
		end

	gs = [function (zᵢ)
		vcat(dynamics_constraint(zᵢ), make_ic_constraint(i)(zᵢ))
	end for i in 1:N]

	return (; H, G, primal_dimension_per_player, Js, gs, θs)
end

function run_nonlq_solver_with_history(H, graph, primal_dimension_per_player, Js, gs, θs, parameter_values, z0_guess = nothing;
	max_iters = 30, tol = 1e-6, verbose = false,
	ls_α_init = 1.0, ls_β = 0.5, ls_c₁ = 1e-4, max_ls_iters = 10,
	to = TimerOutput(), backend = SymbolicTracingUtils.SymbolicsBackend(),
	preoptimization_info = nothing)

	N = nv(graph)
	reverse_topological_order = reverse(topological_sort(graph))

	out_all_augmented_z_est = nothing

	if isnothing(preoptimization_info)
		@timeit to "[Non-LQ Solver][Preoptimization]" begin
			preoptimization_info = preoptimize_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs, θs; backend, to, verbose)
		end
	end

	@timeit to "[Non-LQ Solver][Setup]" begin
		problem_vars = preoptimization_info.problem_vars
		all_variables = preoptimization_info.all_variables
		zs = problem_vars.zs
		λs = problem_vars.λs
		μs = problem_vars.μs

		setup_info = preoptimization_info.setup_info
		πs = setup_info.πs
		K_syms = setup_info.K_syms
		M_fns = setup_info.M_fns
		N_fns = setup_info.N_fns

		mcp_obj = preoptimization_info.mcp_obj
		F = preoptimization_info.F_sym
		linsolver = preoptimization_info.linsolver
		compute_Ks_with_z = preoptimization_info.compute_Ks_with_z

		out_all_augment_variables = preoptimization_info.out_all_augment_variables
		out_all_augmented_z_est = nothing

		z_est = @something(z0_guess, zeros(length(all_variables)))
		if !isnothing(z0_guess)
			println("Using provided initial guess of length $(length(z0_guess)).")
			if length(z0_guess) < length(all_variables)
				@info "Provided initial guess is shorter than required length $(length(all_variables)). Padding with zeros."
				z_est = vcat(z0_guess, zeros(length(all_variables) - length(z0_guess)))
			end
		end

		num_iterations = 0
		convergence_criterion = Inf
		nonlq_solver_status = :BUG_unspecified
		F_eval = similar(F, Float64)

		θ_order = θs isa AbstractDict ? sort(collect(keys(θs))) : 1:length(θs)
		θ_vals_vec = parameter_values isa AbstractDict ? vcat([parameter_values[k] for k in θ_order]...) : vcat([parameter_values[k] for k in θ_order]...)

		function params_for_z(z)
			all_K_evals_vec, _ = compute_K_evals(z, problem_vars, setup_info)
			param_eval_vec = vcat(θ_vals_vec, all_K_evals_vec)
			return param_eval_vec, all_K_evals_vec
		end

		function params_vec_for_z(z)
			param_eval_vec, _ = params_for_z(z)
			return param_eval_vec
		end
	end

	convergence_history = Float64[]

	while true
		@timeit to "[Non-LQ Solver][Iterative Loop]" begin

			@timeit to "[Non-LQ Solver][Iterative Loop][Evaluate K Numerically]" begin
				param_eval_vec, all_K_evals_vec = params_for_z(z_est)
			end

			@timeit to "[Non-LQ Solver][Iterative Loop][Check Convergence]" begin
				mcp_obj.f!(F_eval, z_est, param_eval_vec)
				convergence_criterion = norm(F_eval)
				push!(convergence_history, convergence_criterion)
				@info("Iteration $num_iterations: Convergence criterion = $convergence_criterion")
				if convergence_criterion < tol
					nonlq_solver_status = (num_iterations > 0) ? :solved : :solver_not_run_but_z0_optimal
					break
				end
			end

			if num_iterations >= max_iters
				nonlq_solver_status = :max_iters_reached
				break
			end

			num_iterations += 1

			@timeit to "[Non-LQ Solver][Iterative Loop][Solve Linearized KKT System]" begin
				dz_sol, F_eval_linsolve, linsolve_status = approximate_solve_with_linsolve!(mcp_obj, linsolver, param_eval_vec, z_est; to)
			end

			if linsolve_status != :solved
				nonlq_solver_status = :linear_solver_error
				@warn "Linear solve failed. Exiting prematurely. Return code: $(linsolve_status)"
				break
			end

			@timeit to "[Non-LQ Solver][Iterative Loop][Update Estimate of z]" begin
				α = 1.0
				for ii in 1:10
					next_z_est = z_est .+ α * dz_sol
					param_eval_vec_kp1, all_K_evals_vec_kp1 = params_for_z(next_z_est)

					mcp_obj.f!(F_eval, next_z_est, param_eval_vec_kp1)
					if norm(F_eval) < norm(F_eval_linsolve)
						param_eval_vec = param_eval_vec_kp1
						all_K_evals_vec = all_K_evals_vec_kp1
						break
					else
						α *= 0.5
					end
				end
				next_z_est = z_est .+ α * dz_sol
			end

			z_est = next_z_est

			out_all_augmented_z_est = vcat(z_est, all_K_evals_vec)
		end
	end

	info = (; num_iterations, final_convergence_criterion = convergence_criterion, to)
	return z_est, nonlq_solver_status, info, all_variables, (; πs, zs, λs, μs, θs), (; out_all_augment_variables, out_all_augmented_z_est), convergence_history
end

function perturb_initial_state(x0; rng = Random.default_rng(), scale = 0.05)
	return [x0[i] .+ scale .* randn(rng, length(x0[i])) for i in 1:length(x0)]
end

# =========================
# Multi-run experiment
# =========================
num_runs = 1
max_iters = 200
perturb_scale = 0.1
rng = MersenneTwister(1234)

problem = build_nonlinear_lane_change_problem(x_goal, R, T, Δt)
preoptimization_info = preoptimize_nonlq_solver(
	problem.H,
	problem.G,
	problem.primal_dimension_per_player,
	problem.Js,
	problem.gs,
	problem.θs;
	backend = SymbolicTracingUtils.SymbolicsBackend(),
	verbose = false,
)

histories = Vector{Vector{Float64}}(undef, num_runs)
statuses = Vector{Symbol}(undef, num_runs)
run_lengths = Vector{Int}(undef, num_runs)

for run_id in 1:num_runs
	x0_run = perturb_initial_state(x0_base; rng = rng, scale = perturb_scale)
	parameter_values = x0_run

	_, status, _, _, _, _, history = run_nonlq_solver_with_history(
		problem.H,
		problem.G,
		problem.primal_dimension_per_player,
		problem.Js,
		problem.gs,
		problem.θs,
		parameter_values,
		z0_guess;
		max_iters = max_iters,
		preoptimization_info = preoptimization_info,
	)

	histories[run_id] = history
	statuses[run_id] = status
	run_lengths[run_id] = length(history)
end

# Filter out runs that took too long.
max_allowed_iters = 75
keep_mask = run_lengths .<= max_allowed_iters
histories = histories[keep_mask]
statuses = statuses[keep_mask]
run_lengths = run_lengths[keep_mask]
num_runs_eff = length(histories)
@assert num_runs_eff > 0 "All runs exceeded the max iteration threshold of $max_allowed_iters."

max_len = maximum(length.(histories))
conv_values = fill(NaN, num_runs_eff, max_len)
for (i, h) in enumerate(histories)
	conv_values[i, 1:length(h)] .= h
	if length(h) < max_len
		conv_values[i, (length(h)+1):end] .= h[end]
	end
end

mean_vals = Vector{Float64}(undef, max_len)
std_vals = Vector{Float64}(undef, max_len)
for k in 1:max_len
	vals_k = conv_values[:, k]
	mean_vals[k] = sum(vals_k) / num_runs_eff
	std_vals[k] = num_runs_eff > 1 ? sqrt(sum((vals_k .- mean_vals[k]).^2) / num_runs_eff) : 0.0
end

iters = 0:(max_len-1)

mean_iters = collect(iters)
valid_mean = mean_vals
std_iters = collect(iters)
std_mean = mean_vals
std_vals_valid = std_vals

eps_val = 1e-12

# Log-space statistics for log plot ribbon.
log_vals = log.(max.(conv_values, eps_val))
log_mean_vals = vec(sum(log_vals, dims = 1) ./ num_runs_eff)
log_std_vals = num_runs_eff > 1 ? vec(sqrt.(sum((log_vals .- log_mean_vals').^2, dims = 1) ./ num_runs_eff)) : fill(0.0, max_len)
log_mean = exp.(log_mean_vals)
log_lower = exp.(log_mean_vals .- log_std_vals)
log_upper = exp.(log_mean_vals .+ log_std_vals)

lower_band = max.(std_mean .- std_vals_valid, eps_val)
upper_band = max.(std_mean .+ std_vals_valid, eps_val)

if isempty(mean_iters)
	println("No valid convergence data to plot (all NaN). Skipping plot.")
else
	if length(mean_iters) > 1
		p = plot(
			mean_iters,
			valid_mean,
			label = "mean ± standard deviation",
			xlabel = "iteration",
			ylabel = "convergence criterion",
			yscale = :identity,
			fillalpha = 0.2,
			linewidth = 2,
			color = :blue,
			size = (900, 500),
			legend = false,
			tickfontsize = 12,
		)
	else
		p = plot(
			mean_iters,
			valid_mean,
			label = "mean",
			xlabel = "iteration",
			ylabel = "convergence criterion",
			yscale = :identity,
			linewidth = 2,
			size = (900, 500),
			legend = false,
			tickfontsize = 12,
		)
		scatter!(p, mean_iters, valid_mean, label = false)
	end

	if !isempty(upper_band)
		ylims!(p, (eps_val, maximum(upper_band) * 1.05))
	elseif !isempty(valid_mean)
		ylims!(p, (eps_val, maximum(valid_mean) * 1.05))
	end
	plot_end_iter = max_len - 1 + 10
	xlims!(p, (minimum(mean_iters), plot_end_iter))

	for i in 1:num_runs_eff
		plot!(p, 0:(length(histories[i])-1), histories[i], label = false, color = :gray, alpha = 0.25)
	end

	# Re-draw mean + ribbon on top of the gray traces.
	if length(std_iters) > 1
		plot!(
			p,
			std_iters,
			std_mean,
			label = false,
			ribbon = (std_mean .- lower_band, upper_band .- std_mean),
			fillalpha = 0.2,
			linewidth = 2,
			color = :blue,
		)
	else
		plot!(p, mean_iters, valid_mean, label = "mean")
	end

	savefig(p, "data/nonlinear_convergence_plot.png")
	println("Saved convergence plot to data/nonlinear_convergence_plot.png")

	p_log = plot(
		mean_iters,
		log_mean;
		label = false,
		xlabel = "iteration",
		ylabel = "convergence criterion",
		yscale = :log10,
		ribbon = (log_mean .- log_lower, log_upper .- log_mean),
		fillalpha = 0.2,
		linewidth = 2,
		color = :blue,
		size = (900, 500),
		legend = false,
		tickfontsize = 12,
	)
	for i in 1:num_runs_eff
		plot!(p_log, 0:(length(histories[i])-1), histories[i], label = false, color = :gray, alpha = 0.25)
	end
	# Re-draw mean + ribbon on top of the gray traces.
	plot!(
		p_log,
		mean_iters,
		log_mean;
		label = false,
		ribbon = (log_mean .- log_lower, log_upper .- log_mean),
		fillalpha = 0.2,
		linewidth = 2,
		color = :blue,
	)
	if !isempty(log_upper)
		ylims!(p_log, (eps_val, maximum(log_upper) * 1.05))
	elseif !isempty(valid_mean)
		ylims!(p_log, (eps_val, maximum(valid_mean) * 1.05))
	end
	xlims!(p_log, (minimum(mean_iters), plot_end_iter))
	savefig(p_log, "data/nonlinear_convergence_plot_log.png")
	println("Saved convergence plot to data/nonlinear_convergence_plot_log.png")
end

println("Run statuses (filtered): ", statuses)
