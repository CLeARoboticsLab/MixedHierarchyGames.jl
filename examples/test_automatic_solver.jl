
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


function approximate_solve_with_linsolve!(mcp_obj, linsolver, all_K_evals_vec, z; to = TimerOutput(), verbose = false)
	"""
	Solves the linear system (approximately about point z) defined by the ParametricMCP object using the provided LinearSolve linsolver.

	Parameters
	----------
	mcp_obj: ParametricMCPs.ParametricMCP
		ParametricMCP object defining the system to solve.
	linsolver: LinearSolve.LinearProblem
		LinearSolve object initialized with a linear solve algorithm.
	all_K_evals_vec: Vector{Float64}
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

		# Use all_K_evals_vec in place of parameter θ
		mcp_obj.f!(F, z, all_K_evals_vec)
		mcp_obj.jacobian_z!(∇F, z, all_K_evals_vec)
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

function compute_K_evals(z_est, problem_vars, setup_info; to = TimerOutput())
	"""
	Computes the numeric evaluations of the K matrices for each player, based on the current estimate of all decision variables.

	Parameters
	----------
	z_est (Vector{Float64}) : The current estimate of all decision variables.
	problem_vars (NamedTuple) : A named tuple containing the symbolic variables for each player and the parameter θ.
								At minimum, should contain ws and ys.
	setup_info (NamedTuple) : A named tuple containing (at minimum):
								graph (SimpleDiGraph) : The information structure of the game.
								M_fns (Dict{Int, Any}) : A dictionary of functions for evaluating M matrices for each agent.
								N_fns (Dict{Int, Any}) : A dictionary of functions for evaluating N matrices for each agent.
								π_sizes (Dict{Int, Int}) : A dictionary mapping each player to the size of their KKT condition vector.

	Returns
	-------
	all_K_evals_vec (Vector{Float64}) : A vector of all numeric evaluations of K matrices for each player, concatenated.
	info (NamedTuple) : A named tuple containing:
		M_evals (Dict{Int, Any}) : A dictionary of numeric evaluations of M matrices for each agent.
		N_evals (Dict{Int, Any}) : A dictionary of numeric evaluations of N matrices for each agent.
		K_evals (Dict{Int, Any}) : A dictionary of numeric evaluations of K matrices for each agent.
	"""

	ws = problem_vars.ws
	ys = problem_vars.ys
	M_fns = setup_info.M_fns
	N_fns = setup_info.N_fns
	π_sizes = setup_info.π_sizes
	graph = setup_info.graph

	M_evals = Dict{Int, Any}()
	N_evals = Dict{Int, Any}()
	K_evals = Dict{Int, Any}()

	for ii in reverse(topological_sort(graph))
		# TODO: optimize: we can use one massive augmented vector if we include dummy values for variables we don't have yet.
		# Get the list of symbols we need values for.
		# augmented_variables = all_augmented_variables[ii]
		if has_leader(graph, ii)
			@timeit to "[Compute K Evals] Player $ii" begin
				@timeit to "[Make Augmented z]" begin
					# Create an augmented version using the numerical values that we have (based on z_est and computed follower Ms/Ns).
					augmented_z_est = map(jj -> reshape(K_evals[jj], :), collect(BFSIterator(graph, ii))[2:end]) # skip ii itself
					augmented_z_est = vcat(z_est, augmented_z_est...)
					# augmented_z_est = [z_est; K_evals[jj] for jj in collect(BFSIterator(graph, ii))[2:end]]
				end
				# Produce linearized versions of the current M and N values which can be used.
				@timeit to "[Get Numeric M, N]" begin
					M_evals[ii] = reshape(M_fns[ii](augmented_z_est), π_sizes[ii], length(ws[ii]))
					N_evals[ii] = reshape(N_fns[ii](augmented_z_est), π_sizes[ii], length(ys[ii]))
				end

				@timeit to "[Solve for K]" begin
					K_evals[ii] = M_evals[ii] \ N_evals[ii]
				end
			end
		else
			M_evals[ii] = nothing
			N_evals[ii] = nothing
			K_evals[ii] = nothing
		end
	end

	# Uncomment for timing.
	# show(to)

	# Make a vector of all K_evals for use in ParametricMCP.
	all_K_evals_vec = vcat(map(ii -> reshape(@something(K_evals[ii], Float64[]), :), 1:nv(graph))...)
	# augmented_z_est = vcat(z_est, all_K_evals_vec)
	return all_K_evals_vec, (; M_evals, N_evals, K_evals, to)
end

function armijo_backtracking_linesearch(mcp_obj, compute_Ks_with_z, z_est, dz_sol; to = TimerOutput(), α_init = 1.0, β = 0.5, c₁ = 1e-4, max_ls_iters = 5)
	"""
	Performs an Armijo backtracking line search to find an appropriate step size for updating the estimate of decision variables.

	Parameters
	----------
	mcp_obj: MCPObject
		The MCP object containing the function and gradient information.
	compute_Ks_with_z: Function
		A function that computes the K matrices given the current estimate of decision variables z.
	z_est: Vector{Float64}
		Current estimate of all decision variables.
	dz_sol: Vector{Float64}
		Proposed update direction for the decision variables.
		Proposed update direction for the decision variables.
	α_init: Float64 (default: 1.0)
		Initial step size.
	β: Float64 (default: 0.5)
		Step size reduction factor.
	c₁: Float64 (default: 1e-4)
		Armijo condition parameter.
	max_ls_iters: Int (default: 10)
		Maximum number of line search iterations.

	Returns
	-------
	step_size: Float64
		The determined step size for updating the decision variables.
	success: Bool
		Whether a suitable step size was found within the maximum iterations.
	"""

	zk = copy(z_est)
	p = dz_sol

	α = α_init
	∇F = similar(mcp_obj.jacobian_z!.result_buffer, Float64)
	F_size = size(∇F, 1)
	F = zeros(F_size)

	@timeit to "[Line Search][Iteration Setup]" begin
		# Compute the right hand side of the Armijo condition.
		@timeit to "[Line Search][Iteration Setup][Eval K]" begin
			K_vec0, _ = compute_Ks_with_z(zk)
			all_Ks_vec_kp1 = K_vec0
		end
		@timeit to "[Line Search][Iteration Setup][Eval F]" begin
			mcp_obj.f!(F, zk, K_vec0)
		end
		@timeit to "[Line Search][Iteration Setup][Eval F grad]" begin
			mcp_obj.jacobian_z!(∇F, zk, K_vec0)
		end

		armijo_F0_norm = norm(F)^2
		armijo_∇F0_term = 2 * c₁ * (∇F' * F)' * p
	end
	success = false

	for kk in 1:max_ls_iters
		@timeit to "[Line Search][Iteration Loop]" begin
			# Evaluate merit function at new point.
			zkp1 = zk .+ α * dz_sol
			@timeit to "[Line Search][Iteration Setup][Eval K]" begin
				all_Ks_vec_kp1 = compute_Ks_with_z(zkp1)
			end
			@timeit to "[Line Search][Iteration Setup][Eval F]" begin
				mcp_obj.f!(F, zkp1, all_Ks_vec_kp1) # No parameters
			end

			# This term uses the chain rule to compute the derivative of the merit function ϕ(z + α p) wrt to α.
			DF_kp1 = ∇F' * F
			if norm(F)^2 <= armijo_F0_norm + α * armijo_∇F0_term
				success = true
				break
			else
				α *= β
			end
		end
	end

	return α, success
end

function preoptimize_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs; backend = SymbolicTracingUtils.SymbolicsBackend(), to = TimerOutput(), verbose = false)
	"""
	Precomputes and sets up the necessary components for solving a non-LQ Stackelberg hierarchy game using a linear quasi-policy approximation approach.

	Parameters
	----------
	H (Int) : The number of planning stages (e.g., 1 for open-loop, T for more).
	graph (SimpleDiGraph) : The information structure of the game at each stage, defined as a directed graph.
	primal_dimension_per_player (Vector{Int}) : The dimension of each player's decision variable.
	Js (Dict{Int, Function}) : A dictionary mapping player indices to their objective functions
							   accepting each player's decision variables, z₁, z₂, ..., zₙ, and the parameter θ.
	gs (Vector{Function}) : A vector of equality constraint functions for each player, accepting only that
						  	player's decision variable.
	backend (SymbolicTracingUtils.SymbolicsBackend, optional) : The symbolic backend to use for variable creation (default: SymbolicTracingUtils.SymbolicsBackend()).
	to (TimerOutput, optional) : TimerOutput object for profiling (default: new TimerOutput()).
	verbose (Bool, optional) : Whether to print verbose output (default: false).

	Returns
	-------
	preoptimization_info (NamedTuple) : A named tuple containing:
		- problem_vars: NamedTuple of problem variables (zs, λs, μs, θ, ws, ys).
		- setup_info: NamedTuple of setup information (graph, M_fns, N_fns, π_sizes).
		- mcp_obj: ParametricMCP object.
		- linsolver: LinearSolve object.
		- compute_Ks_with_z: Function to compute K matrices given z.
		- F_sym: Symbolic representation of the MCP function vector.
		- all_variables: A vector of all symbolic variables used in the problem.
		- out_all_augment_variables: A named tuple containing additional symbolic variables used in the linearized approximation for each player.
		- to: TimerOutput object used for profiling.
		- backend: The symbolic backend used for preoptimization.
	"""
	N = nv(graph) # number of players

	# Construct symbols for each player's decision variables.
	@timeit to "Variable Setup" begin
		problem_vars = setup_problem_variables(H, graph, primal_dimension_per_player, gs; backend, verbose)
		all_variables = problem_vars.all_variables
		zs = problem_vars.zs
		λs = problem_vars.λs
		μs = problem_vars.μs
		θ = problem_vars.θ
		ws = problem_vars.ws
		ys = problem_vars.ys
	end

	@timeit to "Precomputation of KKT Jacobians" begin
		out_all_augment_variables, setup_info = setup_approximate_kkt_solver(graph, Js, zs, λs, μs, gs, ws, ys, θ, all_variables, backend; to = TimerOutput(), verbose = false)
		K_syms = setup_info.K_syms
		πs = setup_info.πs
		M_fns = setup_info.M_fns
		N_fns = setup_info.N_fns

		all_K_syms_vec = vcat(map(ii -> reshape(@something(K_syms[ii], Symbolics.Num[]), :), 1:N)...)
		π_sizes = setup_info.π_sizes

		# Set up a function to evaluate the K matrices using only z.
		function compute_Ks_with_z(z)
			all_K_evals_vec, (; to) = compute_K_evals(z, problem_vars, setup_info; to)
			return all_K_evals_vec
		end
	end

	@timeit to "Linear Solver Initialization" begin
		F_size = sum(values(π_sizes))
		linear_solve_algorithm = LinearSolve.UMFPACKFactorization()
		linsolver = init(LinearProblem(spzeros(F_size, F_size), zeros(F_size)), linear_solve_algorithm)
	end

	symbolic_type = eltype(all_variables)

	@timeit to "[Linear Solve] Setup ParametricMCP" begin
		# Final MCP vector: leader stationarity + leader constraints + follower KKT
		F_sym = Vector{symbolic_type}([
			vcat(collect(values(πs))...)..., # KKT conditions of all players
		])

		z̲ = fill(-Inf, length(F_sym));
		z̅ = fill(Inf, length(F_sym))

		# Form mcp via ParametricMCP initialization.
		println(length(all_variables), " variables, ", length(F_sym), " conditions")
		mcp_obj = ParametricMCPs.ParametricMCP(F_sym, all_variables, all_K_syms_vec, z̲, z̅; compute_sensitivities = false)
	end

	return (; problem_vars, setup_info, mcp_obj, F_sym, linsolver, compute_Ks_with_z, all_variables, out_all_augment_variables, to, backend)
end


function run_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs, z0_guess = nothing;
	parameter_value = 1e-5, max_iters = 30, tol = 1e-6, verbose = false,
	ls_α_init = 1.0, ls_β = 0.5, ls_c₁ = 1e-4, max_ls_iters = 10,
	to = TimerOutput(), backend = SymbolicTracingUtils.SymbolicsBackend(),
	preoptimization_info = nothing)
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
								If max_iters = 0, then we only evaluate the KKT conditions at the initial guess and
								the resulting status is set to either :max_iters_reached or :solver_not_run_but_z0_optimal.
								If the status :BUG_unspecified is returned, it indicates that the solver is escaping through
								an unspecified route, which should not happen.
	tol (Float64, optional) : Tolerance for convergence (default: 1e-6).
	verbose (Bool, optional) : Whether to print verbose output (default: false).
	ls_α_init (Float64, optional) : Initial step size for line search (default: 1.0).
	ls_β (Float64, optional) : Step size reduction factor for line search (default: 0.5).
	ls_c₁ (Float64, optional) : Armijo condition parameter for line search (default: 1e-4).
	max_ls_iters (Int, optional) : Maximum number of line search iterations (default: 10).
	preoptimization_info (NamedTuple, optional) : If provided, uses this precomputed information to skip the preoptimization step.
												Should contain at least:
												- problem_vars: NamedTuple of problem variables (zs, λs, μs, θ, ws, ys).
												- setup_info: NamedTuple of setup information (graph, M_fns, N_fns, π_sizes).
												- mcp_obj: ParametricMCP object.
												- linsolver: LinearSolve object.
												- compute_Ks_with_z: Function to compute K matrices given z.
												- F_sym: Symbolic representation of the MCP function vector.
												- all_variables: A vector of all symbolic variables used in the problem.
												- out_all_augment_variables: A named tuple containing additional symbolic variables used in the linearized approximation for each player.
												- to: TimerOutput object used for profiling.
												- backend: The symbolic backend used for preoptimization.

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

	out_all_augmented_z_est = nothing

	if isnothing(preoptimization_info)
		# If we don't have preoptimization info, then run the preoptimization step now.
		preoptimization_info = preoptimize_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs; backend, to, verbose)
	end

	# Unpack preoptimization info - variables.
	problem_vars = preoptimization_info.problem_vars
	all_variables = preoptimization_info.all_variables
	zs = problem_vars.zs
	λs = problem_vars.λs
	μs = problem_vars.μs
	θ = problem_vars.θ
	ws = problem_vars.ws
	ys = problem_vars.ys

	# Unpack preoptimization info - Symbolic KKT conditions and evaluation functions.
	setup_info = preoptimization_info.setup_info
	πs = setup_info.πs
	K_syms = setup_info.K_syms
	M_fns = setup_info.M_fns
	N_fns = setup_info.N_fns

	# Set up solver objects and functions.
	mcp_obj = preoptimization_info.mcp_obj
	F = preoptimization_info.F_sym
	linsolver = preoptimization_info.linsolver
	compute_Ks_with_z = preoptimization_info.compute_Ks_with_z

	# Set up variables for augmented in/output.
	out_all_augment_variables = preoptimization_info.out_all_augment_variables
	out_all_augmented_z_est = nothing

	# Initial guess for primal and dual variables.
	z_est = @something(z0_guess, zeros(length(all_variables)))

	# Set up variables used for tracking number of iterations, convergence, and status in solver loop.
	num_iterations = 0

	# These (nonsensical) values should never be returned or it indicates a bug.
	convergence_criterion = Inf
	nonlq_solver_status = :BUG_unspecified
	F_eval = similar(F, Float64)

	# Run the iteration loop indefinitely, until we satisfy one of the termination conditions.
	# 1. The solution has converged.
	# 2. The max number of iterations is reached.
	# The algorithm always at least checks if the provided guess is a solution.
	while true
		@timeit to "[Non-LQ Solver][Iterative Loop]" begin

			# Compute the numeric K_evals for each player based on the current guess for z.
			@timeit to "[Non-LQ Solver][Iterative Loop][Evaluate K Numerically]" begin
				all_K_evals_vec, _ = compute_K_evals(z_est, problem_vars, setup_info)
			end

			# Check termination condition 1: convergence before starting the next iteration.
			@timeit to "[Non-LQ Solver][Iterative Loop][Check Convergence]" begin
				mcp_obj.f!(F_eval, z_est, all_K_evals_vec)
				convergence_criterion = norm(F_eval)
				println("Iteration $num_iterations: Convergence criterion = $convergence_criterion")
				verbose && println("Iteration $num_iterations: Convergence criterion = $convergence_criterion")
				if convergence_criterion < tol
					# Handle the case where max_iters is set to 0 by checking whether the guess is optimal and then returning.
					nonlq_solver_status = (num_iterations > 0) ? :solved : :solver_not_run_but_z0_optimal
					break
				end
			end

			# Check termination condition 2: max number of iterations reached.
			if num_iterations >= max_iters
				nonlq_solver_status = :max_iters_reached
				break
			end

			# Update the number of iterations after we check for convergence of the previous iteration.
			num_iterations += 1

			# TODO: F_eval is compute multiple times in each loop (in this call, above, linesearch). Optimize if needed.
			@timeit to "[Non-LQ Solver][Iterative Loop][Solve Linearized KKT System]" begin
				dz_sol, F_eval_linsolve, linsolve_status = approximate_solve_with_linsolve!(mcp_obj, linsolver, all_K_evals_vec, z_est; to)
			end

			# If there is a linear solver failure, then exit the loop and return a failure status.
			if linsolve_status != :solved
				nonlq_solver_status = :linear_solver_error
				@warn "Linear solve failed. Exiting prematurely. Return code: $(linsolve_status)"
				break
			end

			# Update the estimate.
			# TODO: Using a constant step size. Add a line search here since we see oscillation.
			@timeit to "[Non-LQ Solver][Iterative Loop][Update Estimate of z]" begin
				α = 1.0
				for ii in 1:10
					next_z_est = z_est .+ α * dz_sol

					# This line recomputes K at each step size, which is correct but inefficient. Uncomment if we want it.
					# all_Ks_vec_kp1 = compute_Ks_with_z(next_z_est)

					# Check if the merit function decreases; currently, we use an outdated version of K.
					mcp_obj.f!(F_eval, next_z_est, all_K_evals_vec)
					if norm(F_eval) < norm(F_eval_linsolve)
						break
					else
						α *= 0.5
					end
				end
				# α = 1. / (num_iterations+1) # Diminishing step size
				# α, _ = armijo_backtracking_linesearch(mcp_obj, compute_Ks_with_z, z_est, dz_sol; to, α_init=ls_α_init, β=ls_β, c₁=ls_c₁, max_ls_iters=max_ls_iters)
				next_z_est = z_est .+ α * dz_sol
			end

			# Update the current estimate.
			z_est = next_z_est

			# Compile all numerical values that we used into one large vector (for this iteration only).
			# TODO: Compile for multiple iterations later.
			out_all_augmented_z_est = vcat(z_est, all_K_evals_vec)
		end
	end

	if verbose
		show(to)
	end

	show(to)

	# TODO: Redo to add more general output.
	info = (; num_iterations, final_convergence_criterion = convergence_criterion)
	z_est, nonlq_solver_status, info, all_variables, (; πs, zs, λs, μs, θ), (; out_all_augment_variables, out_all_augmented_z_est)
end


# TODO: Add a new function that only runs the non-lq solver.
function solve_nonlq_game_example(H, graph, primal_dimension_per_player, Js, gs; z0_guess = nothing, parameter_value = 1e-5, max_iters = 30, tol = 1e-6, verbose = false)
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
								If max_iters = 0, then we only evaluate the KKT conditions at the initial guess and
								the resulting status is set to either :max_iters_reached or :solver_not_run_but_z0_optimal.
								If the status :BUG_unspecified is returned, it indicates that the solver is escaping through
								an unspecified route, which should not happen.
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
	to = TimerOutput()
	preoptimization_info = preoptimize_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs; verbose)
	z_sol, status, info, all_variables, vars, augmented_vars = run_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs, z0_guess; preoptimization_info, parameter_value, max_iters, tol = 1e-3, verbose, to)
	return z_sol, status, info, all_variables, vars, augmented_vars
end

function compare_lq_and_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs; parameter_value = 1e-5, verbose = false)
	"""
	We can only run this comparison on an LQ game, or it will error.
	"""

	# Run our solver through the run_lq_solver call.
	z_sol_nonlq, status_nonlq, info_nonlq, all_variables, (; πs, zs, λs, μs, θ), all_augmented_vars = run_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs; tol = 1e-3, parameter_value, verbose)

	println("Non-LQ solver status after $(info_nonlq.num_iterations) iterations: $(status_nonlq)")

	lq_vars = setup_problem_variables(H, graph, primal_dimension_per_player, gs; verbose)
	all_lq_variables = lq_vars.all_variables
	lq_zs = lq_vars.zs
	lq_λs = lq_vars.λs
	lq_μs = lq_vars.μs
	lq_θ = lq_vars.θ
	lq_ws = lq_vars.ws
	lq_ys = lq_vars.ys

	to1 = TimerOutput()
	@timeit to1 "LQ KKT Condition Gen" begin
		lq_πs, lq_Ms, lq_Ns, _ = get_lq_kkt_conditions(graph, Js, lq_zs, lq_λs, lq_μs, gs, lq_ws, lq_ys, lq_θ)
	end
	@timeit to1 "LQ KKT Condition Solve" begin
		z_sol_lq, status_lq = lq_game_linsolve(lq_πs, all_lq_variables, lq_θ, parameter_value; verbose)
		z_sol_lq = nothing
		status_lq = :not_implemented
	end
	show(to1)

	# TODO: Ensure that the solutions are the same for an LQ example.
	@assert isapprox(z_sol_nonlq, z_sol_lq, atol = 1e-4)
	@show "Are they the same? > $(isapprox(z_sol_nonlq, z_sol_lq, atol = 1e-4))"
	@show status_lq

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

	# Normalize x0 to Vector{Vector{Float64}} to support inputs from Python (which often pass a Matrix)
	x0_vecs = if x0 isa AbstractMatrix
		[collect(@view x0[i, :]) for i in 1:size(x0, 1)]
	elseif x0 isa AbstractVector{<:AbstractVector}
		x0
	else
		error("x0 must be a Matrix or a Vector of Vectors")
	end

	# Number of players in the game
	N = 3

	# Set up the information structure.
	# This defines a stackelberg chain with three players, where P2 is the leader of P1 and P3, which
	# are Nash with each other.
	G = SimpleDiGraph(N);
	add_edge!(G, 2, 1); # P2 -> P1
	add_edge!(G, 2, 3); # P2 -> P3


	H = 1
	Hp1 = H+1 # number of planning stages is 1 for OL game.

	# Helper function
	flatten(vs) = collect(Iterators.flatten(vs))

	# Initial sizing of various dimensions.
	T = 10 # time horizon
	Δt = 0.5 # time step
	state_dimension = 2 # player 1,2,3's state dimension
	control_dimension = 2 # player 1,2,3's control dimension

	# Additional dimension computations.
	x_dim = state_dimension * (T+1)
	u_dim = control_dimension * (T+1)
	aggre_state_dimension = x_dim * N
	aggre_control_dimension = u_dim * N
	total_dimension = aggre_state_dimension + aggre_control_dimension
	primal_dimension_per_player = x_dim + u_dim

	# Print dimension information.
	println("Number of players: $N")
	println("Number of Stages: $H (OL = 1; FB > 1)")
	println("Time Horizon (# steps): $T")
	println("Step period: Δt = $(Δt)s")
	println("Dimension per player: $(primal_dimension_per_player)")
	println("Total primal dimension: $total_dimension")


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
		return x1 - x0_vecs[i]
	end

	# In this game, each player has the same dynamics constraint.
	dynamics_constraint(zᵢ) =
		mapreduce(vcat, 1:T) do t
			dynamics(zᵢ, t)
		end
	gs = [function (zᵢ)
		vcat(dynamics_constraint(zᵢ), make_ic_constraint(i)(zᵢ))
	end for i in 1:N]

	parameter_value = 1e-5
	# z_sol_nonlq, status_nonlq, z_sol_lq, status_lq, info_nonlq, all_variables, vars, all_augmented_vars = compare_lq_and_nonlq_solver(H, G, primal_dimension_per_player, Js, gs; parameter_value, verbose)
	z_sol_nonlq, status_nonlq, info_nonlq, all_variables, vars, all_augmented_vars = solve_nonlq_game_example(H, G, primal_dimension_per_player, Js, gs; parameter_value, verbose)
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

	# # Plot the trajectories.
	# plot_player_trajectories(xs1, xs2, xs3, T, Δt, verbose)

	###################OUTPUT: next state, current control ######################
	next_state = Vector{Vector{Float64}}()
	curr_control = Vector{Vector{Float64}}()
	(; xs, us) = unflatten_trajectory(z₁_sol, state_dimension, control_dimension)
	push!(next_state, xs[2]) # next state of player 1
	push!(curr_control, us[1]) # current control of player 1
	if verbose
		println("P1 (x,u) solution : ($xs, $us)")
		println("P1 Objective: $(Js[1](z₁_sol, z₂_sol, z₃_sol, 0))")
	end
	(; xs, us) = unflatten_trajectory(z₂_sol, state_dimension, control_dimension)
	push!(next_state, xs[2]) # next state of player 2
	push!(curr_control, us[1]) # current control of player 2
	if verbose
		println("P2 (x,u) solution : ($xs, $us)")
		println("P2 Objective: $(Js[2](z₁_sol, z₂_sol, z₃_sol, 0))")
	end
	(; xs, us) = unflatten_trajectory(z₃_sol, state_dimension, control_dimension)
	push!(next_state, xs[2]) # next state of player 3
	push!(curr_control, us[1]) # current control of player 3
	if verbose
		println("P3 (x,u) solution : ($xs, $us)")
		println("P3 Objective: $(Js[3](z₁_sol, z₂_sol, z₃_sol, 0))")
	end

	return next_state, curr_control
	# next_state: [ [x1_next], [x2_next], [x3_next] ] = [ [-0.0072, 1.7970], [1.7925, 3.5889], [5.4159, 7.2201] ] where xi_next = [ pⁱ_x, pⁱ_y]
	# curr_control: [ [u1_curr], [u2_curr], [u3_curr] ] = [ [-0.0144, -0.4060], [-0.4150, -0.8222], [-1.1683, -1.5598] ] where ui_curr = [ vⁱ_x, vⁱ_y]
end

function nplayer_hierarchy_navigation_bicycle_dynamics(x0; verbose = false)
	"""
	Navigation function for a multi-player hierarchy game. Players are modeled with bicycle dynamics in 2D space, 
		with objectives to reach certain sets of game states.

	Parameters
	----------
	x0 (Vector{Vector{Float64}}) : A vector of initial conditions for each player, where each initial condition is a vector [px, py].
	verbose (Bool, optional) : Whether to print verbose output (default: false).
	"""

	# Normalize x0 to Vector{Vector{Float64}} to support inputs from Python (which often pass a Matrix)
	x0_vecs = if x0 isa AbstractMatrix
		[collect(@view x0[i, :]) for i in 1:size(x0, 1)]
	elseif x0 isa AbstractVector{<:AbstractVector}
		x0
	else
		error("x0 must be a Matrix or a Vector of Vectors")
	end

	# Number of players in the game
	N = 3

	# Set up the information structure.
	# This defines a stackelberg chain with three players, where P2 is the leader of P1 and P3, which
	# are Nash with each other.
	G = SimpleDiGraph(N);
	add_edge!(G, 2, 1); # P2 -> P1
	add_edge!(G, 2, 3); # P2 -> P3


	H = 1
	Hp1 = H+1 # number of planning stages is 1 for OL game.

	# Helper function
	flatten(vs) = collect(Iterators.flatten(vs))

	# Initial sizing of various dimensions.
	T = 10 # time horizon
	Δt = 0.5 # time step
	state_dimension = 4 # player 1,2,3's state dimension (x = [px, py, ψ, v])
	control_dimension = 2 # player 1,2,3's control dimension (u = [a, δ])

	# Additional dimension computations.
	x_dim = state_dimension * (T+1)
	u_dim = control_dimension * (T+1)
	aggre_state_dimension = x_dim * N
	aggre_control_dimension = u_dim * N
	total_dimension = aggre_state_dimension + aggre_control_dimension
	primal_dimension_per_player = x_dim + u_dim

	# Print dimension information.
	println("Number of players: $N")
	println("Number of Stages: $H (OL = 1; FB > 1)")
	println("Time Horizon (# steps): $T")
	println("Step period: Δt = $(Δt)s")
	println("Dimension per player: $(primal_dimension_per_player)")
	println("Total primal dimension: $total_dimension")

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

	function bicycle_dynamics(z, t; Δt = Δt, L = 1.0)
		# Kinematic bicycle dynamics (nonlinear)
		# State:   x = [x, y, ψ, v]
		# Control: u = [a, δ]
		# Euler forward discretization:
		#   x_{t+1} = x_t + Δt * [ v*cosψ, v*sinψ, (v/L)*tanδ, a ]

		(; xs, us)  = unflatten_trajectory(z, state_dimension, control_dimension)
		x_t         = xs[t]       # = [x, y, ψ, v]
		u_t         = us[t]       # = [a, δ]
		x_tp1       = xs[t+1]
		x, y, ψ, v = x_t
		a, δ       = u_t

		# One-step prediction under bicycle model
		xdot = v * cos(ψ)
		ydot = v * sin(ψ)
		psidot = (v / L) * tan(δ)
		vdot = a

		x_pred = x_t .+ Δt .* [xdot, ydot, psidot, vdot]

		return x_tp1 - x_pred
	end

	# 2-D double integrator: state = [x, y, vx, vy], control = [ax, ay]
	function dynamics_double_integrator_2D(z, t; Δt = Δt)
		(; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)

		x_t   = xs[t]
		u_t   = us[t]
		x_tp1 = xs[t+1]

		x, y, vx, vy = x_t
		ax, ay = u_t

		x_next  = x + Δt*vx + 0.5*Δt^2*ax
		y_next  = y + Δt*vy + 0.5*Δt^2*ay
		vx_next = vx + Δt*ax
		vy_next = vy + Δt*ay

		x_pred = [x_next, y_next, vx_next, vy_next]
		return x_tp1 - x_pred
	end

	# Set up the equality constraints for each player.
	make_ic_constraint(i) = function (zᵢ)
		(; xs, us) = unflatten_trajectory(zᵢ, state_dimension, control_dimension)
		x1 = xs[1]
		return x1 - x0_vecs[i]
	end

	# In this game, each player has the same dynamics constraint.
	dynamics_constraint(zᵢ) =
		mapreduce(vcat, 1:T) do t
			bicycle_dynamics(zᵢ, t)
			# dynamics_double_integrator_2D(zᵢ, t)
		end
	gs = [function (zᵢ)
		vcat(dynamics_constraint(zᵢ), make_ic_constraint(i)(zᵢ))
	end for i in 1:N]

	parameter_value = 1e-5
	# z_sol_nonlq, status_nonlq, z_sol_lq, status_lq, info_nonlq, all_variables, vars, all_augmented_vars = compare_lq_and_nonlq_solver(H, G, primal_dimension_per_player, Js, gs; parameter_value, verbose)
	z_sol_nonlq, status_nonlq, info_nonlq, all_variables, vars, all_augmented_vars = solve_nonlq_game_example(H, G, primal_dimension_per_player, Js, gs; parameter_value, verbose)
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

	# # Plot the trajectories.
	plot_player_trajectories(xs1, xs2, xs3, T, Δt, verbose)

	###################OUTPUT: next state, current control ######################
	next_state = Vector{Vector{Float64}}()
	curr_control = Vector{Vector{Float64}}()
	(; xs, us) = unflatten_trajectory(z₁_sol, state_dimension, control_dimension)
	push!(next_state, xs[2]) # next state of player 1
	push!(curr_control, us[1]) # current control of player 1
	if verbose
		println("P1 (x,u) solution : ($xs, $us)")
		println("P1 Objective: $(Js[1](z₁_sol, z₂_sol, z₃_sol, 0))")
	end
	(; xs, us) = unflatten_trajectory(z₂_sol, state_dimension, control_dimension)
	push!(next_state, xs[2]) # next state of player 2
	push!(curr_control, us[1]) # current control of player 2
	if verbose
		println("P2 (x,u) solution : ($xs, $us)")
		println("P2 Objective: $(Js[2](z₁_sol, z₂_sol, z₃_sol, 0))")
	end
	(; xs, us) = unflatten_trajectory(z₃_sol, state_dimension, control_dimension)
	push!(next_state, xs[2]) # next state of player 3
	push!(curr_control, us[1]) # current control of player 3
	if verbose
		println("P3 (x,u) solution : ($xs, $us)")
		println("P3 Objective: $(Js[3](z₁_sol, z₂_sol, z₃_sol, 0))")
	end

	return next_state, curr_control
	# next_state: [ [x1_next], [x2_next], [x3_next] ] = [ [-0.0072, 1.7970], [1.7925, 3.5889], [5.4159, 7.2201] ] where xi_next = [ pⁱ_x, pⁱ_y]
	# curr_control: [ [u1_curr], [u2_curr], [u3_curr] ] = [ [-0.0144, -0.4060], [-0.4150, -0.8222], [-1.1683, -1.5598] ] where ui_curr = [ vⁱ_x, vⁱ_y]
end
