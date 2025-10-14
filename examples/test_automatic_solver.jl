
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes
using Graphs
using InvertedIndices
using LinearAlgebra: I, norm, pinv, Diagonal, rank
using LinearSolve: LinearSolve, LinearProblem, init, solve!
using Plots
using TimerOutputs: TimerOutput, @timeit
using TrajectoryGamesBase: unflatten_trajectory
using SciMLBase: SciMLBase
using SparseArrays: spzeros
using Symbolics
using SymbolicTracingUtils

include("graph_utils.jl")
include("make_symbolic_variables.jl")
include("solve_kkt_conditions.jl")
include("evaluate_results.jl")
include("general_kkt_construction.jl")

include("automatic_solver.jl")


function approximate_solve_with_linsolve!(mcp_obj, linsolver, K_evals_vec, z; to=TimerOutput(), verbose = false)
	"""
	Solves the linear system (approximately about point z) defined by the ParametricMCP object using the provided LinearSolve linsolver.

	Parameters
	----------
	mcp_obj: ParametricMCPs.ParametricMCP
		ParametricMCP object defining the system to solve.
	linsolver: LinearSolve.LinearProblem
		LinearSolve object initialized with a linear solve algorithm.
	K_evals_vec: Vector{Float64}
		Vector of numeric values for the parameters in the ParametricMCP object.
	z: Vector{Float64}
		Vector of numeric values for the decision variables in the ParametricMCP object, about which we linearize the system.
	to: TimerOutput (default: new TimerOutput())
		TimerOutput object for performance profiling.
	verbose: Bool (default: false)
		Whether to print verbose output.

	Returns
	-------
	z_sol: Vector{Float64}
		Solution vector for all variables.
	F: Vector{Float64}
		Evaluated function vector at the solution.
	linsolve_status: Symbol
		Status of the linear solve. (:solved or :failed).
	"""

	@timeit to "[Linear Solve] ParametricMCP Setup" begin
		∇F = mcp_obj.jacobian_z!.result_buffer

		F_size = size(∇F, 1)
		F = zeros(F_size)
		δz = zeros(F_size)
	end

	@timeit to "[Linear Solve] Jacobian Eval + Problem Definition" begin
		linsolve_status = :solved

		# Use K_evals_vec in place of parameter θ
		mcp_obj.f!(F, z, K_evals_vec)
		mcp_obj.jacobian_z!(∇F, z, K_evals_vec)
		linsolver.A = ∇F
		linsolver.b = -F
	end

	@timeit to "[Linear Solve] Call to Solver" begin
		solution = solve!(linsolver)
	end

	if !SciMLBase.successful_retcode(solution) &&
	   (solution.retcode !== SciMLBase.ReturnCode.Default)
		verbose &&
			@warn "Linear solve failed. Exiting prematurely. Return code: $(solution.retcode)"
		linsolve_status = :failed
	else
		z_sol = solution.u
	end

	return z_sol, F, linsolve_status
end

# TODO: Write helpers that setup the variables ahead of time and once so it's not repeated.
function run_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs, z0_guess=nothing; 
						  parameter_value = 1e-5, max_iters = 30, tol = 1e-6, verbose = false)
	"""
	Solves a non-LQ Stackelberg hierarchy game using a linear quasi-policy approximation approach.

	Parameters
	----------
	H (Int) : The number of planning stages (e.g., 1 for open-loop, T for more).
	graph (SimpleDiGraph) : The information structure of the game at each stage, defined as a directed graph.
	primal_dimension_per_player (Vector{Int}) : The dimension of each player's decision variable.
	Js (Dict{Int, Function}) : A dictionary mapping player indices to their objective functions 
							   accepting each player's decision variables, z₁, z₂, ..., zₙ, and the parameter θ.
	gs (Vector{Function}) : A vector of equality constraint functions for each player, accepting only that
						  	player's decision variable.
	z0_guess (Vector{Float64}, optional) : An optional initial guess for the decision variables.
										   If not provided, defaults to a zero vector.
	parameter_value (Float64, optional) : Numeric value to substitute for the symbolic parameter θ (default: 1e-5).
	max_iters (Int, optional) : Maximum number of iterations for the solver (default: 30).
	tol (Float64, optional) : Tolerance for convergence (default: 1e-6).
	verbose (Bool, optional) : Whether to print verbose output (default: false).

	Returns
	-------
	z_sol (Vector{Float64}) : The approximate solution vector containing all players' decision variables.
	status (Symbol) : The status of the solver (:solved, :max_iters_reached, or :failed).
	info (Dict) : A dictionary containing information about the solver's performance, including
				  the number of iterations and final convergence criterion.
	all_variables (Vector{Num}) : A vector of all symbolic variables used in the problem.
	vars (NamedTuple) : A named tuple containing the symbolic variables for each player and the parameter θ.
						(zs, λs, μs, θ).
	augmented_vars (NamedTuple) : A named tuple containing additional symbolic variables (and numeric evaluations of them) 
									used in the linearized approximation for each player.
									(out_all_augment_variables, out_all_augmented_z_est).
	"""


	N = nv(graph) # number of players
	reverse_topological_order = reverse(topological_sort(graph))

	#TODO: Add line search for non-LQ case
	# Main Solver loop
	nonlq_solver_status = :max_iters_reached
	iters = 0

	# Create a TimerOutput object
	to = TimerOutput()

	# Construct symbolic backend for each player's decision variables.
	backend = SymbolicTracingUtils.SymbolicsBackend()

	# Construct symbols for each player's decision variables.
	@timeit to "Variable Setup" begin
		(; all_variables, zs, λs, μs, θ, ws, ys) = setup_problem_variables(H, graph, primal_dimension_per_player, gs; backend, verbose)
	end

	@timeit to "Precomputation of KKT Jacobians" begin
		out_all_augment_variables, setup_info = setup_approximate_kkt_solver(graph, Js, zs, λs, μs, gs, ws, ys, θ, all_variables, backend; to=TimerOutput(), verbose = false)
		K_syms = setup_info.K_syms
		πs = setup_info.πs
		M_fns = setup_info.M_fns
		N_fns = setup_info.N_fns

		all_vectorized_Ks = vcat(map(ii -> reshape(@something(K_syms[ii], Symbolics.Num[]), :), 1:N)...)
		π_sizes = setup_info.π_sizes

		out_all_augmented_z_est = nothing
	end

	@timeit to "Linear Solver Initialization" begin
		F_size = sum(values(π_sizes))
		linear_solve_algorithm = LinearSolve.UMFPACKFactorization()
		linsolver = init(LinearProblem(spzeros(F_size, F_size), zeros(F_size)), linear_solve_algorithm)
	end

	symbolic_type = eltype(all_variables)

	@timeit to "[Linear Solve] Setup ParametricMCP" begin
		# Final MCP vector: leader stationarity + leader constraints + follower KKT
		F = Vector{symbolic_type}([
			vcat(collect(values(πs))...)..., # KKT conditions of all players
		])

		# TODO: Set up F and Lagrangian L as a call function of z and K.

		z̲ = fill(-Inf, length(F));
		z̅ = fill(Inf, length(F))

		# Form mcp via ParametricMCP initialization.
		println(length(all_variables), " variables, ", length(F), " conditions")
		mcp_obj = ParametricMCPs.ParametricMCP(F, all_variables, all_vectorized_Ks, z̲, z̅; compute_sensitivities = false)
	end

	# Initial guess for primal and dual variables.
	z_est = @something(z0_guess, zeros(length(all_variables)))
	πs_est = nothing

	# Set up variables for convergence checking.
	convergence_criterion = Inf

	# Run this loop until convergence or max iterations reached.
	while convergence_criterion > tol && iters <= max_iters
		@timeit to "Iterative Loop" begin
			iters += 1

			@timeit to "Iterative Loop[KKT Conditions with z0]" begin
				@timeit to "[KKT Conditions][Non-Leaf][Numeric][Evaluate M]" begin

					M_evals = Dict{Int, Any}()

					N_evals = Dict{Int, Any}()
					K_evals = Dict{Int, Any}()
					for ii in reverse_topological_order
						# TODO: Can be made more efficient if needed.
						# πⁱ has size num constraints + num primal variables of i AND its followers.
						# π_sizes[ii] = length(gs[ii](zs[ii]))
						# for jj in BFSIterator(G, ii) # loop includes ii itself.
						# 	π_sizes[ii] += length(zs[jj])
						# end

						# TODO: optimize: we can use one massive augmented vector if we include dummy values for variables we don't have yet.
						# Get the list of symbols we need values for.
						# augmented_variables = all_augmented_variables[ii]

						if has_leader(graph, ii)
							# Create an augmented version using the numerical values that we have (based on z_est and computed follower Ms/Ns).
							augmented_z_est = map(jj -> reshape(K_evals[jj], :), collect(BFSIterator(graph, ii))[2:end]) # skip ii itself
							augmented_z_est = vcat(z_est, augmented_z_est...)
							
							# Produce linearized versions of the current M and N values which can be used.
							M_evals[ii] = reshape(M_fns[ii](augmented_z_est), π_sizes[ii], length(ws[ii]))
							N_evals[ii] = reshape(N_fns[ii](augmented_z_est), π_sizes[ii], length(ys[ii]))

							K_evals[ii] = M_evals[ii] \ N_evals[ii]
						else
							M_evals[ii] = nothing
							N_evals[ii] = nothing
							K_evals[ii] = nothing
						end
					end
				end


				# Compute the numeric vectors (of type Float64, generally).
				for ii in 1:N
					K_evals[ii] = @something(K_evals[ii], Float64[])
				end
				all_vectorized_Kevals = vcat(map(ii -> reshape(@something(K_evals[ii], Float64[]), :), 1:N)...)
			end

			@timeit to "Iterative Loop[Linear Solve]" begin
				dz_sol, F, linsolve_status = approximate_solve_with_linsolve!(mcp_obj, linsolver, all_vectorized_Kevals, z_est; to)
				if linsolve_status != :solved
					status = :failed
					@warn "Linear solve failed. Exiting prematurely. Return code: $(linsolve_status)"
					break
				end
			end

			# Update the estimate.
			# TODO: Using a constant step size. Add a line search here since we see oscillation.
			α = 1.
			next_z_est = z_est .+ α * dz_sol

			# Update the convergence criterion.
			convergence_criterion = norm(F)
			println("Convergence: ", convergence_criterion)
			if convergence_criterion < tol
				nonlq_solver_status = :solved
			end

			# Update the current estimate.
			z_est = next_z_est
			πs_est = πs

			# Compile all numerical values that we used into one large vector (for this iteration only).
			# TODO: Compile for multiple iterations later.
			out_all_augmented_z_est = vcat(z_est, vcat(map(ii -> reshape(K_evals[ii], :), 1:N))...)
		end
	end

	if verbose
		show(to)
	end

	# TODO: Redo to add more general output.
	info = Dict("iterations" => iters, "final convergence_criterion" => convergence_criterion)
	πs = πs_est
	z_est, nonlq_solver_status, info, all_variables, (; πs, zs, λs, μs, θ), (;out_all_augment_variables, out_all_augmented_z_est)
end


# TODO: Add a new function that only runs the non-lq solver.
function compare_lq_and_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs; parameter_value = 1e-5, verbose = false)

	# Run our solver through the run_lq_solver call.
	z_sol_nonlq, status_nonlq, info_nonlq, all_variables, (; πs, zs, λs, μs, θ), all_augmented_vars = run_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs; parameter_value, verbose)
	
	println("Non-LQ solver status after $(info_nonlq["iterations"]) iterations: $(status_nonlq)")

	lq_vars = setup_problem_variables(H, graph, primal_dimension_per_player, gs; verbose)
	all_lq_variables = lq_vars.all_variables
	lq_zs = lq_vars.zs
	lq_λs = lq_vars.λs
	lq_μs = lq_vars.μs
	lq_θ = lq_vars.θ
	lq_ws = lq_vars.ws
	lq_ys = lq_vars.ys

	# lq_πs, lq_Ms, lq_Ns, _ = get_lq_kkt_conditions(graph, Js, lq_zs, lq_λs, lq_μs, gs, lq_ws, lq_ys, lq_θ)
	# z_sol_lq, status_lq = lq_game_linsolve(lq_πs, all_lq_variables, lq_θ, parameter_value; verbose)
	z_sol_lq = nothing
	status_lq = :not_implemented

	# TODO: Ensure that the solutions are the same for an LQ example.
	# @assert isapprox(z_sol_nonlq, z_sol_lq, atol = 1e-4)
	# @show "Are they the same? > $(isapprox(z_sol_nonlq, z_sol_lq, atol = 1e-4))"
	# @show status_lq

	z_sol_nonlq, status_nonlq, z_sol_lq, status_lq, info_nonlq, all_variables, (; πs, zs, λs, μs, θ), all_augmented_vars
end


######### INPUT: Initial conditions ##########################
x0 = [
	[0.0; 2.0], # [px, py]
	[2.0; 4.0],
	[6.0; 8.0],
]
###############################################################

# Main body of algorithm implementation for hardware. Will restructure as needed.
function nplayer_hierarchy_navigation(x0; verbose = false)
	"""
	Navigation function for a multi-player hierarchy game. Players are modeled as double integrators in 2D space, 
		with objectives to reach certain sets of game states.

	Parameters
	----------
	x0 (Vector{Vector{Float64}}) : A vector of initial conditions for each player, where each initial condition is a vector [px, py].
	verbose (Bool, optional) : Whether to print verbose output (default: false).
	"""

	# Number of players in the game
	N = 3

	# Set up the information structure.
	# This defines a stackelberg chain with three players, where P1 is the leader of P2, and P1+P2 are leaders of P3.
	G = SimpleDiGraph(N);
	add_edge!(G, 2, 1); # P1 -> P2
	add_edge!(G, 2, 3); # P2 -> P3


	H = 1
	Hp1 = H+1 # number of planning stages is 1 for OL game.

	# Helper function
	flatten(vs) = collect(Iterators.flatten(vs))

	# Initial sizing of various dimensions.
	T = 3 # time horizon
	state_dimension = 2 # player 1,2,3's state dimension
	control_dimension = 2 # player 1,2,3's control dimension

	# Additional dimension computations.
	x_dim = state_dimension * (T+1)
	u_dim = control_dimension * (T+1)
	aggre_state_dimension = x_dim * N
	aggre_control_dimension = u_dim * N
	total_dimension = aggre_state_dimension + aggre_control_dimension
	primal_dimension_per_player = x_dim + u_dim


	#### Player Objectives ####
	# Player 1's objective function: P1 wants to get close to P2's final position 
	# considering only its own control effort.
	function J₁(z₁, z₂, z₃, θ)
		(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
		xs¹, us¹ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
		xs², us² = xs, us
		0.5*sum((xs¹[end] .- xs²[end]) .^ 2) + 0.05*sum(sum(u .^ 2) for u in us¹)
	end

	# Player 2's objective function: P2 wants P1 and P3 to get to the origin
	function J₂(z₁, z₂, z₃, θ)
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
		xs³, us³ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
		xs², us² = xs, us
		(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
		xs¹, us¹ = xs, us
		sum((0.5*(xs¹[end] .+ xs³[end])) .^ 2) + 0.05*sum(sum(u .^ 2) for u in us²)
	end

	# Player 3's objective function: P3 wants to get close to P2's final position considering its own and P2's control effort.
	function J₃(z₁, z₂, z₃, θ)
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
		xs³, us³ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
		xs², us² = xs, us
		0.5*sum((xs³[end] .- xs²[end]) .^ 2) + 0.05*sum(sum(u³ .^ 2) for u³ in us³) + 0.05*sum(sum(u² .^ 2) for u² in us²)
	end

	Js = Dict{Int, Any}(
		1 => J₁,
		2 => J₂,
		3 => J₃,
	)


	#### player individual dynamics ####
	Δt = 0.5 # time step

	# Dynamics are the only constraints (for now).
	function dynamics(z, t)
		(; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
		x = xs[t]
		u = us[t]
		xp1 = xs[t+1]
		# rows 3:4 for p2 in A, and columns 3:4 for p2 in B when using the full stacked system
		# but since A is I and B is block-diagonal by design, you can just write:
		return xp1 - x - Δt*u
	end

	# Set up the equality constraints for each player.
	make_ic_constraint(i) = function (zᵢ)
		(; xs, us) = unflatten_trajectory(zᵢ, state_dimension, control_dimension)
		x1 = xs[1]
		return x1 - x0[i]
	end

	# In this game, each player has the same dynamics constraint.
	dynamics_constraint(zᵢ) = mapreduce(vcat, 1:T) do t dynamics(zᵢ, t) end
	gs = [function (zᵢ) vcat(dynamics_constraint(zᵢ), make_ic_constraint(i)(zᵢ)) end for i in 1:N]

	parameter_value = 1e-5
	z_sol_nonlq, status_nonlq, z_sol_lq, status_lq, info_nonlq, all_variables, vars, all_augmented_vars = compare_lq_and_nonlq_solver(H, G, primal_dimension_per_player, Js, gs; parameter_value, verbose)
	z_sol = z_sol_nonlq
	(; πs, zs, λs, μs, θ) = vars
	(; out_all_augment_variables, out_all_augmented_z_est) = all_augmented_vars

	z₁ = zs[1]
	z₂ = zs[2]
	z₃ = zs[3]
	z₁_sol = z_sol[1:length(z₁)]
	z₂_sol = z_sol[(length(z₁)+1):(length(z₁)+length(z₂))]
	z₃_sol = z_sol[(length(z₁)+length(z₂)+1):(length(z₁)+length(z₂)+length(z₃))]

	# TODO: Update this to work with the new formulation.
	# Evaluate the KKT residuals at the solution to check solution quality.
	z_sols = [z₁_sol, z₂_sol, z₃_sol]
	evaluate_kkt_residuals(πs, out_all_augment_variables, out_all_augmented_z_est, θ, parameter_value; verbose = verbose)
	# evaluate_kkt_residuals(πs, all_variables, z_sol, θ, parameter_value; verbose = verbose)

	# Reconstruct trajectories from solutions
	xs1, _ = unflatten_trajectory(z₁_sol, state_dimension, control_dimension)
	xs2, _ = unflatten_trajectory(z₂_sol, state_dimension, control_dimension)
	xs3, _ = unflatten_trajectory(z₃_sol, state_dimension, control_dimension)

	# Plot the trajectories.
	plot_player_trajectories(xs1, xs2, xs3, T, Δt, verbose)

	###################OUTPUT: next state, current control ######################
	next_state = Vector{Vector{Float64}}()
	curr_control = Vector{Vector{Float64}}()
	(; xs, us) = unflatten_trajectory(z₁_sol, state_dimension, control_dimension)
	push!(next_state, xs[2]) # next state of player 1
	push!(curr_control, us[1]) # current control of player 1
	println("P1 (x,u) solution : ($xs, $us)")
	println("P1 Objective: $(Js[1](z₁_sol, z₂_sol, z₃_sol, 0))")
	(; xs, us) = unflatten_trajectory(z₂_sol, state_dimension, control_dimension)
	push!(next_state, xs[2]) # next state of player 2
	push!(curr_control, us[1]) # current control of player 2
	println("P2 (x,u) solution : ($xs, $us)")
	println("P2 Objective: $(Js[2](z₁_sol, z₂_sol, z₃_sol, 0))")
	(; xs, us) = unflatten_trajectory(z₃_sol, state_dimension, control_dimension)
	push!(next_state, xs[2]) # next state of player 3
	push!(curr_control, us[1]) # current control of player 3
	println("P3 (x,u) solution : ($xs, $us)")
	println("P3 Objective: $(Js[3](z₁_sol, z₂_sol, z₃_sol, 0))")

	return next_state, curr_control
	# next_state: [ [x1_next], [x2_next], [x3_next] ] = [ [-0.0072, 1.7970], [1.7925, 3.5889], [5.4159, 7.2201] ] where xi_next = [ pⁱ_x, pⁱ_y]
	# curr_control: [ [u1_curr], [u2_curr], [u3_curr] ] = [ [-0.0144, -0.4060], [-0.4150, -0.8222], [-1.1683, -1.5598] ] where ui_curr = [ vⁱ_x, vⁱ_y]
end
