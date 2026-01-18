
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
using JLD2

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

function _build_augmented_z_est(ii, z_est, K_evals, graph, follower_order_cache, buffer_cache)
	followers = get!(follower_order_cache, ii) do
		collect(BFSIterator(graph, ii))[2:end]
	end

	aug_len = length(z_est)
	for jj in followers
		kj = K_evals[jj]
		aug_len += isnothing(kj) ? 0 : length(kj)
	end

	buf = get!(buffer_cache, ii) do
		Vector{Float64}(undef, aug_len)
	end
	if length(buf) != aug_len
		resize!(buf, aug_len)
	end

	copyto!(buf, 1, z_est, 1, length(z_est))
	offset = length(z_est) + 1
	for jj in followers
		kj = K_evals[jj]
		if isnothing(kj)
			continue
		end
		flat = reshape(kj, :)
		copyto!(buf, offset, flat, 1, length(flat))
		offset += length(flat)
	end

	buf
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

	# Cache follower ordering and augmented buffers to reduce allocations.
	follower_order_cache = Dict{Int, Vector{Int}}()
	buffer_cache = Dict{Int, Vector{Float64}}()

	for ii in reverse(topological_sort(graph))
		# TODO: optimize: we can use one massive augmented vector if we include dummy values for variables we don't have yet.
		# Get the list of symbols we need values for.
		if has_leader(graph, ii)
			@timeit to "[Compute K Evals] Player $ii" begin
				@timeit to "[Make Augmented z]" begin
					# Create an augmented version using the numerical values that we have (based on z_est and computed follower Ms/Ns).
					# For small numbers of variables, this caching may not help much with runtime.
					# TODO: Use this only when the number of variables reaches a sufficient size where it will matter.
					augmented_z_est = _build_augmented_z_est(ii, z_est, K_evals, graph, follower_order_cache, buffer_cache)
				end
				# Produce linearized versions of the current M and N values which can be used.
				@timeit to "[Get Numeric M, N]" begin
					# Main.@infiltrate
					M_evals[ii] = reshape(M_fns[ii](augmented_z_est), π_sizes[ii], length(ws[ii]))
					N_evals[ii] = reshape(N_fns[ii](augmented_z_est), π_sizes[ii], length(ys[ii]))
				end

				@timeit to "[Solve for K]" begin
					zi_size = length(problem_vars.zs[ii])
					extractor = hcat(I(zi_size), zeros(zi_size, length(ws[ii]) - zi_size))
					K_evals[ii] = M_evals[ii] \ N_evals[ii]
					# Compute the policy ϕ_i = -M_i^{-1} N_i
					ϕⁱ = -extractor * K_evals[ii]
					println("Player $ii policy ϕ[$ii] to be embedded: ", norm(ϕⁱ))
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

function preoptimize_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs, θs; backend = SymbolicTracingUtils.SymbolicsBackend(), to = TimerOutput(), verbose = false)
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
	θs (Dict{Int, Vector{Num}}) : The parameters symbols.
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
		- π_sizes_trimmed: A dictionary mapping each player to the size of their trimmed KKT condition vector.
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
		ws = problem_vars.ws
		ys = problem_vars.ys
	end

	@timeit to "Precomputation of KKT Jacobians" begin
		out_all_augment_variables, setup_info = setup_approximate_kkt_solver(graph, Js, zs, λs, μs, gs, ws, ys, θs, all_variables, backend; to = TimerOutput(), verbose = false)
		K_syms = setup_info.K_syms
		πs = setup_info.πs
		M_fns = setup_info.M_fns
		N_fns = setup_info.N_fns
		π_sizes = setup_info.π_sizes

		all_K_syms_vec = vcat(map(ii -> reshape(@something(K_syms[ii], Symbolics.Num[]), :), 1:N)...)
		θ_order = θs isa AbstractDict ? sort(collect(keys(θs))) : 1:length(θs)
		θ_syms_flat = vcat([θs[i] for i in θ_order]...)
		all_param_syms_vec = vcat(θ_syms_flat, all_K_syms_vec)

		# Set up a function to evaluate the K matrices using only z.
		function compute_Ks_with_z(z)
			all_K_evals_vec, (; to) = compute_K_evals(z, problem_vars, setup_info; to)
			return all_K_evals_vec
		end
	end

	symbolic_type = eltype(all_variables)

	@timeit to "[Linear Solve] Setup ParametricMCP" begin
		# Final MCP vector: leader stationarity + leader constraints + follower KKT
		πs_solve = strip_policy_constraints(πs, graph, zs, gs)
		π_sizes_trimmed = Dict(ii => length(πs_solve[ii]) for ii in keys(πs_solve))
		π_order = sort(collect(keys(πs_solve)))
		F_sym = Symbolics.Num.(vcat([πs_solve[i] for i in π_order]...))  # flattened in deterministic order

		z̲ = fill(-Inf, length(F_sym));
		z̅ = fill(Inf, length(F_sym))

		# Form mcp via ParametricMCP initialization.
		@info "$(length(all_variables)) variables, $(length(F_sym)) conditions"
		params_syms_vec = Symbolics.Num.(all_param_syms_vec)
		mcp_obj = ParametricMCPs.ParametricMCP(F_sym, all_variables, params_syms_vec, z̲, z̅; compute_sensitivities = false)
	end

	@timeit to "Linear Solver Initialization" begin
		F_size = length(F_sym)
		linear_solve_algorithm = LinearSolve.UMFPACKFactorization()
		linsolver = init(LinearProblem(spzeros(F_size, F_size), zeros(F_size)), linear_solve_algorithm)
	end

	return (; problem_vars, setup_info, mcp_obj, F_sym, linsolver, compute_Ks_with_z, all_variables, out_all_augment_variables, π_sizes_trimmed, to, backend)
end


function run_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs, θs, parameter_values, z0_guess = nothing;
	max_iters = 30, tol = 1e-6, verbose = false,
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
												- π_sizes_trimmed: A dictionary mapping each player to the size of their trimmed KKT condition vector.
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
		@timeit to "[Non-LQ Solver][Preoptimization]" begin
			# If we don't have preoptimization info, then run the preoptimization step now.
			preoptimization_info = preoptimize_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs, θs; backend, to, verbose)
		end
	end

	@timeit to "[Non-LQ Solver][Setup]" begin
		# Unpack preoptimization info - variables.
		problem_vars = preoptimization_info.problem_vars
		all_variables = preoptimization_info.all_variables
		zs = problem_vars.zs
		λs = problem_vars.λs
		μs = problem_vars.μs

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
		if !isnothing(z0_guess)
			println("Using provided initial guess of length $(length(z0_guess)).")
			if length(z0_guess) < length(all_variables)
				@info "Provided initial guess is shorter than required length $(length(all_variables)). Padding with zeros."
				z_est = vcat(z0_guess, zeros(length(all_variables) - length(z0_guess)))
			end
		end

		# Set up variables used for tracking number of iterations, convergence, and status in solver loop.
		num_iterations = 0

		# These (nonsensical) values should never be returned or it indicates a bug.
		convergence_criterion = Inf
		nonlq_solver_status = :BUG_unspecified
		F_eval = similar(F, Float64)

		θ_order = θs isa AbstractDict ? sort(collect(keys(θs))) : 1:length(θs)
		θ_vals_vec = parameter_values isa AbstractDict ? vcat([parameter_values[k] for k in θ_order]...) : vcat([parameter_values[k] for k in θ_order]...)

		# Helper to compute the parameter values (θ, K) for a given z to pass into ParametricMCPs.
		function params_for_z(z)
			all_K_evals_vec, _ = compute_K_evals(z, problem_vars, setup_info)
			param_eval_vec = vcat(θ_vals_vec, all_K_evals_vec)
			return param_eval_vec, all_K_evals_vec
		end
	end

	# Run the iteration loop indefinitely, until we satisfy one of the termination conditions.
	while true
		@timeit to "[Non-LQ Solver][Iterative Loop]" begin

			@timeit to "[Non-LQ Solver][Iterative Loop][Evaluate K Numerically]" begin
				param_eval_vec, all_K_evals_vec = params_for_z(z_est)
			end

			@timeit to "[Non-LQ Solver][Iterative Loop][Check Convergence]" begin
				mcp_obj.f!(F_eval, z_est, param_eval_vec)
				convergence_criterion = norm(F_eval)
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
					# Recompute params at the trial point, which is correct but inefficient, and gets the parameters.
					param_eval_vec_kp1, all_K_evals_vec_kp1 = params_for_z(next_z_est)

					# Check if the merit function decreases and update K; currently, we use an outdated version of K.
					mcp_obj.f!(F_eval, next_z_est, param_eval_vec_kp1)
					if norm(F_eval) < norm(F_eval_linsolve)
						param_eval_vec = param_eval_vec_kp1
						all_K_evals_vec = all_K_evals_vec_kp1
						break
					else
						α *= 0.5
					end
				end
				# α = 1. / (num_iterations+1) # Diminishing step size
				# α, _ = armijo_backtracking_linesearch(mcp_obj, compute_Ks_with_z, z_est, dz_sol; to, α_init = ls_α_init, β = ls_β, c₁ = ls_c₁, max_ls_iters = max_ls_iters)
				next_z_est = z_est .+ α * dz_sol
			end

			# Update the current estimate.
			z_est = next_z_est

			# Compile all numerical values that we used into one large vector (for this iteration only).
			out_all_augmented_z_est = vcat(z_est, all_K_evals_vec)
		end
	end

	# TODO: Redo to add more general output.
	info = (; num_iterations, final_convergence_criterion = convergence_criterion, to)
	z_est, nonlq_solver_status, info, all_variables, (; πs, zs, λs, μs, θs), (; out_all_augment_variables, out_all_augmented_z_est)
end


# TODO: Add a new function that only runs the non-lq solver.
function solve_nonlq_game_example(H, graph, primal_dimension_per_player, Js, gs, θs, parameter_values;
	z0_guess = nothing, max_iters = 30, tol = 1e-6, include_preoptimization_timing = false, verbose = false,
	backend = SymbolicTracingUtils.SymbolicsBackend(), multiple_runs = 1)
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
	θs (Dict{Int, Vector{Num}}) : The parameters symbols.
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
	# Initialize timing output so we always measure the entire time, but not with all granularity.
	to = TimerOutput()

	# Get preoptimized info.
	preoptimization_info = preoptimize_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs, θs; backend = backend, verbose = verbose, to = to)

	# Optional warmup to trigger compilation before timed solve.
	let
		z0 = zeros(length(preoptimization_info.all_variables))
		θ_order = θs isa AbstractDict ? sort(collect(keys(θs))) : 1:length(θs)
		θ_vals_vec = parameter_values isa AbstractDict ? vcat([parameter_values[k] for k in θ_order]...) : vcat(parameter_values...)
		K0, _ = compute_K_evals(z0, preoptimization_info.problem_vars, preoptimization_info.setup_info)
		param0 = vcat(θ_vals_vec, K0)
		mcp = preoptimization_info.mcp_obj
		Fbuf = similar(preoptimization_info.F_sym, Float64)
		Jbuf = mcp.jacobian_z!.result_buffer
		mcp.f!(Fbuf, z0, param0)
		mcp.jacobian_z!(Jbuf, z0, param0)
	end

	# If specified, use the preoptimization timing information. Else, make a new one.
	to = include_preoptimization_timing ? preoptimization_info.to : TimerOutput()

	# Run the non-LQ solver.
	z_sol, status, info, all_variables, vars, augmented_vars = nothing, nothing, nothing, nothing, nothing, nothing
	for i in 1:multiple_runs
		@timeit to "[Call to Non-LQ Solver]" begin
			z_sol, status, info, all_variables, vars, augmented_vars = run_nonlq_solver(
				H, graph, primal_dimension_per_player, Js, gs, θs, parameter_values, z0_guess;
				preoptimization_info = preoptimization_info, backend = backend, max_iters = max_iters,
				tol = 1e-6, verbose = verbose, to = to,
			)
		end
	end

	return z_sol, status, info, all_variables, vars, augmented_vars
end

function compare_lq_and_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs, θs, parameter_values, backend = SymbolicTracingUtils.SymbolicsBackend(); verbose = false)
	"""
	We can only run this comparison on an LQ game, or it will error.
	"""

	# Run our solver through the run_lq_solver call.
	z_sol_nonlq, status_nonlq, info_nonlq, all_variables, (; πs, zs, λs, μs, θs), all_augmented_vars = run_nonlq_solver(
		H, graph, primal_dimension_per_player, Js, gs, θs, parameter_values;
		backend = backend, tol = 1e-3, verbose = verbose,
	)

	verbose && @info("Non-LQ solver status after $(info_nonlq.num_iterations) iterations: $(status_nonlq)")

	lq_vars = setup_problem_variables(H, graph, primal_dimension_per_player, gs; verbose)
	all_lq_variables = lq_vars.all_variables
	lq_zs = lq_vars.zs
	lq_λs = lq_vars.λs
	lq_μs = lq_vars.μs
	lq_ws = lq_vars.ws
	lq_ys = lq_vars.ys

	to1 = TimerOutput()
	@timeit to1 "LQ KKT Condition Gen" begin
		lq_πs, lq_Ms, lq_Ns, _ = get_lq_kkt_conditions(graph, Js, lq_zs, lq_λs, lq_μs, gs, lq_ws, lq_ys, θs)
	end
	@timeit to1 "LQ KKT Condition Solve" begin
		lq_πs_solve = strip_policy_constraints(lq_πs, graph, lq_zs, gs)
		z_sol_lq, status_lq = lq_game_linsolve(lq_πs_solve, all_lq_variables, θs, parameter_values; verbose)
	end
	info_lq = (; πs = lq_πs)
	if verbose
		show(to1)
	end

	# Ensure that the solutions are the same for the LQ solver and the non-LQ solver on an LQ example.
	@assert isapprox(z_sol_nonlq, z_sol_lq, atol = 1e-4)
	if verbose
		is_same_solution = isapprox(z_sol_nonlq, z_sol_lq, atol = 1e-4)
		@info "LQ status: $status_lq \nDo the LQ and non-LQ solver produce the same solution? > $is_same_solution)"
	end

	z_sol_nonlq, status_nonlq, z_sol_lq, status_lq, info_nonlq, info_lq, all_variables, (; πs, zs, λs, μs, θs), all_augmented_vars
end


######### INPUT: Initial conditions ##########################
x0 = [
	[0.0; 2.0], # [px, py]
	[2.0; 4.0],
	[6.0; 8.0],
]
###############################################################

# Main body of algorithm implementation for hardware. Will restructure as needed.
function nplayer_hierarchy_navigation(x0; run_lq=false, verbose=false, show_timing_info=false, strip_policy_constraints_eval=true)
	"""
	Navigation function for a multi-player hierarchy game. Players are modeled as double integrators in 2D space, 
		with objectives to reach certain sets of game states.

	Parameters
	----------
	x0 (Vector{Vector{Float64}}) : A vector of initial conditions for each player, where each initial condition is a vector [px, py].
	verbose (Bool, optional) : Whether to print verbose output (default: false).
	"""

	# TODO: Use the initial condition in the initial guess.
	# Normalize x0 to Vector{Vector{Float64}} to support inputs from Python (which often pass a Matrix)
	x0_vecs = if x0 isa AbstractMatrix
		[collect(@view x0[i, :]) for i in 1:size(x0, 1)]
	elseif x0 isa AbstractVector{<:AbstractVector}
		x0
	else
		error("x0 must be a Matrix or a Vector of Vectors")
	end

	# Set up the problem.
	T = 3
	Δt = 0.5
	N, G, H, problem_dims, Js, gs, θs, backend = get_three_player_openloop_lq_problem(T, Δt; verbose)

	primal_dimension_per_player = problem_dims.primal_dimension_per_player
	state_dimension = problem_dims.state_dimension
	control_dimension = problem_dims.control_dimension

	# Modify P2's objective function to be nonlinear unless we are running an LQ game.
	if !run_lq
		# Player 2's objective function: P2 wants P1 and P3 to get to the origin
		function J₂_quartic(z₁, z₂, z₃, θi)
			(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
			xs³, us³ = xs, us
			(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
			xs², us² = xs, us
			(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
			xs¹, us¹ = xs, us
			sum((0.5*(xs¹[end] .+ xs³[end])) .^ 4) + 0.05*sum(sum(u .^ 2) for u in us²)
		end
		Js[2] = J₂_quartic
	end

	# Print dimension information.
	@info "Problem dimensions:\n" *
		  "  Number of players: $N\n" *
		  "  Number of Stages: $H (OL = 1; FB > 1)\n" *
		  "  Time Horizon (# steps): $T\n" *
		  "  Step period: Δt = $(Δt)s\n" *
		  "  Dimension per player: $(primal_dimension_per_player)\n" *
		  "  Total primal dimension: $(problem_dims.total_dimension)"

	# Solve the game using our non-LQ solver. Use initial states as the parameter value vector.
	parameter_values = x0_vecs
	if run_lq
		z_sol_nonlq, status_nonlq, z_sol_lq, status_lq, info_nonlq, info_lq, all_variables, vars, all_augmented_vars = compare_lq_and_nonlq_solver(H, G, primal_dimension_per_player, Js, gs, θs, parameter_values, backend; verbose)
		if show_timing_info
			show(info_nonlq.to)
		end
	else
		z_sol_nonlq, status_nonlq, info_nonlq, all_variables, vars, all_augmented_vars = solve_nonlq_game_example(H, G, primal_dimension_per_player, Js, gs, θs, parameter_values; verbose)
		println("Non-LQ solver status after $(info_nonlq.num_iterations) iterations: $(status_nonlq)")
		if show_timing_info
			show(info_nonlq.to)
		end
	end
	z_sol = z_sol_nonlq
	(; πs, zs, λs, μs, θs) = vars
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

	# Evaluate the solution against the KKT conditions (or approximate KKT conditions for non-LQ).
	if run_lq
		πs_eval_lq = strip_policy_constraints_eval ? strip_policy_constraints(info_lq.πs, G, zs, gs) : info_lq.πs
		evaluate_kkt_residuals(πs_eval_lq, all_variables, z_sol_lq, θs, x0_vecs; verbose = true)
	end
	πs_eval = strip_policy_constraints_eval ? strip_policy_constraints(πs, G, zs, gs) : πs
	evaluate_kkt_residuals(πs_eval, out_all_augment_variables, out_all_augmented_z_est, θs, x0_vecs; verbose = true)


	# Plot the trajectories.
	plot_player_trajectories(z_sols, T, Δt, problem_dims)

	# Print solution information.
	verbose && print_solution_info(z_sols, Js, problem_dims)

	# Report objective value for each agent at the solved trajectories.
	# Note: in this example, there are no parameters in the objectives, so we pass `nothing`.
	costs = [Js[i](z_sols[1], z_sols[2], z_sols[3], nothing) for i in 1:N]
	println()
	@info "Agent costs" costs=costs


	###################OUTPUT: next state, current control ######################
	next_state, curr_control = construct_output(z_sols, problem_dims; verbose = verbose)

	return next_state, curr_control
	# next_state: [ [x1_next], [x2_next], [x3_next] ] = [ [-0.0072, 1.7970], [1.7925, 3.5889], [5.4159, 7.2201] ] where xi_next = [ pⁱ_x, pⁱ_y]
	# curr_control: [ [u1_curr], [u2_curr], [u3_curr] ] = [ [-0.0144, -0.4060], [-0.4150, -0.8222], [-1.1683, -1.5598] ] where ui_curr = [ vⁱ_x, vⁱ_y]
end


function construct_output(z_sols, problem_dims; verbose = false)
	"""
	Constructs the output format for the navigation function.
	"""

	next_state = Vector{Vector{Float64}}()
	curr_control = Vector{Vector{Float64}}()

	for (i, z_sol) in enumerate(z_sols)
		(; xs, us) = unflatten_trajectory(z_sol, problem_dims.state_dimension, problem_dims.control_dimension)
		push!(next_state, xs[2]) # next state of player i
		push!(curr_control, us[1]) # current control of player i

		if verbose
			println("P$i (x,u) solution : ($xs, $us)")
			println("P$i Objective: $(Js[i](z_sols..., nothing))")
		end
	end

	return next_state, curr_control
end



function nplayer_hierarchy_navigation_nonlinear_dynamics(x0, x_goal, z0_guess, R, T, Δt; max_iters = 50, verbose = false, strip_policy_constraints_eval=true)
	"""
	Navigation function for a multi-player hierarchy game. Players are modeled with nonlinear (unicycle, bicycle) dynamics in 2D space, 
		with objectives to reach certain sets of game states.

	Parameters
	----------
	x0 (Vector{Vector{Float64}}) : A vector of initial conditions for each player, where each initial condition is a vector [px, py].
	verbose (Bool, optional) : Whether to print verbose output (default: false).
	strip_policy_constraints_eval (Bool, optional) : Whether to strip policy constraints when evaluating KKT residuals (default: true).
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
	N = 4

	# Set up the information structure.
	G = SimpleDiGraph(N);

	# 1. Comment all below for all-Nash baseline

	# 2. Shallow-tree
	# add_edge!(G, 1, 2); # P1 -> P2
	# add_edge!(G, 1, 3); # P1 -> P3
	# add_edge!(G, 1, 4); # P1 -> P4

	# 3. mixed A
	# add_edge!(G, 1, 2); # P1 -> P2
	# add_edge!(G, 2, 4); # P2 -> P4
	# add_edge!(G, 1, 3); # P1 -> P3

	# 4. Stackelberg chain
	add_edge!(G, 1, 3); # P1 -> P3
	add_edge!(G, 3, 2); # P3 -> P2
	add_edge!(G, 2, 4); # P2 -> P4

	# 5. mixed B
	# add_edge!(G, 1, 2); # P1 -> P2
	# add_edge!(G, 2, 4); # P2 -> P4

	H = 1
	Hp1 = H+1 # number of planning stages is 1 for OL game.

	# Helper function
	flatten(vs) = collect(Iterators.flatten(vs))

	# Initial sizing of various dimensions.
	# T = 10 # time horizon, 10
	# Δt = 0.1 # time step
	state_dimension = 4 # player 1,2,3's state dimension (x = [px, py, ψ, v]) unicycle
	control_dimension = 2 # player 1,2,3's control dimension (u = [a, ω])

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
	# Player 1's objective function: 
	function J₁(z₁, z₂, z₃, z₄, θ)
		(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
		xs¹, us¹ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
		xs², us² = xs, us
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
		xs³, us³ = xs, us
		(; xs, us) = unflatten_trajectory(z₄, state_dimension, control_dimension)
		xs⁴, us⁴ = xs, us

		# ordering = sum(0.5*((xs¹[end] .- x_goal) .^ 2 .+ (xs³[end] .- x_goal) .^ 2))
		control = 10sum(sum(u .^ 2) for u in us¹)
		collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
		velocity = sum((x¹[4] - 2.0)^2 for x¹ in xs¹) # penalize high speeds
		y_deviation = sum((x¹[2]-R)^2 for x¹ in xs¹) # penalize y deviation from R
		zero_heading = sum((x¹[3])^2 for x¹ in xs¹) # penalize heading away from 0

		# Commands to the followers 
		y_deviation_P2 = sum((x²[2]-R)^2 for x² in xs²) # directs the follower P1 to go straight
		zero_heading_P2 = sum((x²[3])^2 for x² in xs²)
		y_deviation_P3 = sum((x³[2]-R)^2 for x³ in xs³[div(T, 2):T]) # directs the follower P3 to go straight
		zero_heading_P3 = sum((x³[3])^2 for x³ in xs³[div(T, 2):T])
		y_deviation_P4 = sum((x⁴[2]-R)^2 for x⁴ in xs⁴) # directs the follower P4 to go straight
		zero_heading_P4 = sum((x⁴[3])^2 for x⁴ in xs⁴)


		# sum((0.5*((xs¹[end] .- x_goal)) .^ 2)) + 0.05*sum(sum(u .^ 2) for u in us²)

		# control + collision + y_deviation + zero_heading +
		# y_deviation_P2 + zero_heading_P2
		control + collision + 5y_deviation + zero_heading + velocity
	end

	# Player 2's objective function: 
	function J₂(z₁, z₂, z₃, z₄, θ)
		(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
		xs¹, us¹ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
		xs², us² = xs, us
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
		xs³, us³ = xs, us
		(; xs, us) = unflatten_trajectory(z₄, state_dimension, control_dimension)
		xs⁴, us⁴ = xs, us

		y_deviation_P4 = sum((x⁴[2]-R)^2 for x⁴ in xs⁴) # directs the follower P4 to go straight
		zero_heading_P4 = sum((x⁴[3])^2 for x⁴ in xs⁴)

		tracking = 0.5*sum((xs¹[end][1:2] .- xs²[end][1:2]) .^ 2) # track only the final position of the leader
		control = sum(sum(u .^ 2) for u in us²)
		collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
		velocity = sum((x²[4] - 2.0)^2 for x² in xs²) # penalize high speeds
		# velocity = sum((xs²[t][4] - xs¹[t][4])^2 for t in 1:T) # penalize high speeds relative to leader
		y_deviation = sum((x²[2]-R)^2 for x² in xs²) # penalize y deviation from R
		zero_heading = sum((x²[3])^2 for x² in xs²) # penalize heading away from 0

		# control + collision + y_deviation + zero_heading + velocity +
		# y_deviation_P4 + zero_heading_P4
		control + collision + 5y_deviation + zero_heading + velocity
	end

	# Player 3's objective function: P3 wants to get close to P2's final position + stay on the circular track for the first half
	function J₃(z₁, z₂, z₃, z₄, θ)
		(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
		xs¹, us¹ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
		xs², us² = xs, us
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
		xs³, us³ = xs, us
		(; xs, us) = unflatten_trajectory(z₄, state_dimension, control_dimension)
		xs⁴, us⁴ = xs, us

		y_deviation_P2 = sum((x²[2]-R)^2 for x² in xs²) # directs the follower P1 to go straight
		zero_heading_P2 = sum((x²[3])^2 for x² in xs²)

		tracking = 10sum((sum(x³[1:2] .^ 2) - R^2)^2 for x³ in xs³[2:div(T, 2)]) #+ 0.5*sum((xs³[end][1:2] .- xs¹[end][1:2]) .^ 2) # track only the final position of the leader
		control = sum(sum(u³ .^ 2) for u³ in us³)
		collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
		velocity = sum((x³[4] - 2.0)^2 for x³ in xs³) # penalize high speeds
		y_deviation = sum((x³[2]-R)^2 for x³ in xs³[div(T, 2):T])
		zero_heading = sum((x³[3])^2 for x³ in xs³[div(T, 2):T])

		# tracking + control + collision + y_deviation + zero_heading + velocity
		tracking + control + collision + 5y_deviation + zero_heading + velocity
	end

	# Player 4's objective function: 
	function J₄(z₁, z₂, z₃, z₄, θ)
		(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
		xs¹, us¹ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
		xs², us² = xs, us
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
		xs³, us³ = xs, us
		(; xs, us) = unflatten_trajectory(z₄, state_dimension, control_dimension)
		xs⁴, us⁴ = xs, us

		tracking = 0.5*sum((xs⁴[end][1:2] .- xs¹[end][1:2]) .^ 2) # track only the final position of the leader
		control = sum(sum(u .^ 2) for u in us⁴)
		collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
		velocity = sum((x⁴[4] - 2.0)^2 for x⁴ in xs⁴) # penalize high speeds
		# velocity = sum((xs⁴[t][4] - xs¹[t][4])^2 for t in 1:T) # penalize high speeds relative to leader
		y_deviation = sum((x⁴[2]-R)^2 for x⁴ in xs⁴) # penalize y deviation from R
		zero_heading = sum((x⁴[3])^2 for x⁴ in xs⁴) # penalize heading away from 0

		control + collision + y_deviation + zero_heading + velocity
	end

	Js = Dict{Int, Any}(
		1 => J₁,
		2 => J₂,
		3 => J₃,
		4 => J₄,
	)

	# Parameter symbols for initial states (one θ vector per player).
	num_params_per_player = fill(state_dimension, N)
	backend = SymbolicTracingUtils.SymbolicsBackend()
	θs = setup_problem_parameter_variables(backend, num_params_per_player; verbose = false)


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

	function unicycle_dynamics(z, t; Δt = Δt)
		# Kinematic unicycle dynamics (nonlinear)
		# State:   x = [x, y, ψ, v]
		# Control: u = [a, ω]      (longitudinal acceleration, yaw rate)
		# Euler forward discretization:
		#   x_{t+1} = x_t + Δt * [ v*cosψ, v*sinψ, ω, a ]

		(; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
		x_t = xs[t]        # = [x, y, ψ, v]
		u_t = us[t]        # = [a, ω]
		x_tp1 = xs[t+1]

		x, y, ψ, v = x_t
		a, ω       = u_t

		# One-step prediction under unicycle model
		xdot   = v * cos(ψ)
		ydot   = v * sin(ψ)
		psidot = ω
		vdot   = a

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

	# Set up the equality constraints for each player using parameterized initial states.
	make_ic_constraint(i) = function (zᵢ)
		(; xs, us) = unflatten_trajectory(zᵢ, state_dimension, control_dimension)
		x1 = xs[1]
		return x1 - θs[i]
	end


	# In this game, each player has the same dynamics constraint.
	dynamics_constraint(zᵢ) =
		mapreduce(vcat, 1:T) do t
			unicycle_dynamics(zᵢ, t)
			# dynamics_double_integrator_2D(zᵢ, t)
		end

	# Combine dynamics and initial condition constraints for each player.
	gs = [function (zᵢ)
		vcat(dynamics_constraint(zᵢ), make_ic_constraint(i)(zᵢ))
	end for i in 1:N]

	parameter_values = x0_vecs
	z_sol_nonlq, status_nonlq, info_nonlq, all_variables, vars, all_augmented_vars = solve_nonlq_game_example(
		H, G, primal_dimension_per_player, Js, gs, θs, parameter_values;
		z0_guess, max_iters, tol = 1e-6, verbose,
		backend = backend,
	)
	(; πs, zs, λs, μs, θs) = vars
	(; out_all_augment_variables, out_all_augmented_z_est) = all_augmented_vars


	z₁ = zs[1]
	z₂ = zs[2]
	z₃ = zs[3]
	z₄ = zs[4]
	z₁_sol = z_sol_nonlq[1:length(z₁)]
	z₂_sol = z_sol_nonlq[(length(z₁)+1):(length(z₁)+length(z₂))]
	z₃_sol = z_sol_nonlq[(length(z₁)+length(z₂)+1):(length(z₁)+length(z₂)+length(z₃))]
	z₄_sol = z_sol_nonlq[(length(z₁)+length(z₂)+length(z₃)+1):(length(z₁)+length(z₂)+length(z₃)+length(z₄))]

	# TODO: Update this to work with the new formulation.
	# Evaluate the KKT residuals at the solution to check solution quality.
	z_sols = [z₁_sol, z₂_sol, z₃_sol, z₄_sol]
	πs_eval = strip_policy_constraints_eval ? strip_policy_constraints(πs, G, zs, gs) : πs
	evaluate_kkt_residuals(πs_eval, out_all_augment_variables, out_all_augmented_z_est, θs, parameter_values; verbose = true)
	# evaluate_kkt_residuals(πs, all_variables, z_sol, θ, parameter_value; verbose = verbose)

	# Reconstruct trajectories from solutions
	xs1, _ = unflatten_trajectory(z₁_sol, state_dimension, control_dimension)
	xs2, _ = unflatten_trajectory(z₂_sol, state_dimension, control_dimension)
	xs3, _ = unflatten_trajectory(z₃_sol, state_dimension, control_dimension)
	xs4, _ = unflatten_trajectory(z₄_sol, state_dimension, control_dimension)

	# Plot the trajectories using the standard helper signature.
	bicycle_problem_dims = (;
		state_dimension,
		control_dimension,
		total_dimension,
		primal_dimension_per_player,
	)
	plot_trajectories_and_distances(xs1, xs2, xs3, xs4, R, T, Δt, verbose)

	###################OUTPUT: next state, current control ######################
	next_state, curr_control = construct_output(z_sols, bicycle_problem_dims; verbose = verbose)

	return next_state, curr_control
	# next_state: [ [x1_next], [x2_next], [x3_next] ] = [ [-0.0072, 1.7970], [1.7925, 3.5889], [5.4159, 7.2201] ] where xi_next = [ pⁱ_x, pⁱ_y]
	# curr_control: [ [u1_curr], [u2_curr], [u3_curr] ] = [ [-0.0144, -0.4060], [-0.4150, -0.8222], [-1.1683, -1.5598] ] where ui_curr = [ vⁱ_x, vⁱ_y]


	# For integration with python code:
	# 1. get strategy 
	# 2. turn into dictionary because you can't send lists 
	# 3. write(sock, JSON3.write(controller_dict) * "\n"), flush(sock) # write to python
	# 4. msg = readline(sock) # Read msg from python
	# 5. data = JSON3.read(String(msg)) 
end


# smooth pairwise penalty for two trajectories
function smooth_collision(xsA, xsB; d_safe = 2.0, α = 20.0, w = 1.0)
	T = length(xsA)
	cost = zero(xsA[1][1])  # symbolic-friendly zero
	d_safe_sq = d_safe^2
	for k in 1:T
		Δp = xsA[k][1:2] .- xsB[k][1:2]
		d_sq = sum(Δp .^ 2)                    # ||pA - pB||^2
		r = d_safe_sq - d_sq                   # positive if too close
		h = (1/α) * log(1 + exp(α * r))        # softplus(r)
		cost += w * h^2
	end
	return cost
end

function smooth_collision_all(xs_all...; d_safe = 2.0, α = 20.0, w = 1.0)
	N = length(xs_all)
	@assert N >= 2 "smooth_collision_all needs at least two players."

	total = 0.0

	# Sum over all unordered pairs (i < j)
	for i in 1:(N-1)
		for j in (i+1):N
			# accumulate cost
			total += 0.1smooth_collision(xs_all[i], xs_all[j]; d_safe, α, w)
		end
	end

	return total
end
