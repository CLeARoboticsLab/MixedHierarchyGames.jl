
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes
using Graphs
using InvertedIndices
using LinearAlgebra: I, norm, pinv, Diagonal, rank
using LinearSolve: LinearSolve, LinearProblem, init, solve!
using Plots
using Symbolics
using SymbolicTracingUtils
using TrajectoryGamesBase: unflatten_trajectory
using SciMLBase: SciMLBase
using TimerOutputs

include("graph_utils.jl")
include("make_symbolic_variables.jl")
include("solve_kkt_conditions.jl")
include("evaluate_results.jl")
include("general_kkt_construction.jl")

include("automatic_solver.jl")


"""                      
∇F(z; ϵ) δz = -F(z; ϵ).
"""
# TODO: Rewrite so that it takes conditions and solves directly instead of iteratively.
function custom_solve_unnecessary_loop(πs, variables, θ, parameter_value; linear_solve_algorithm = LinearSolve.UMFPACKFactorization(), verbose = false)
	symbolic_type = eltype(variables)
	# Final MCP vector: leader stationarity + leader constraints + follower KKT
	F = Vector{symbolic_type}([
		vcat(collect(values(πs))...)..., # KKT conditions of all players
	])

	z̲ = fill(-Inf, length(F));
	z̅ = fill(Inf, length(F))

	# Form mcp via PATH
	parametric_mcp = ParametricMCPs.ParametricMCP(F, variables, [θ], z̲, z̅; compute_sensitivities = false)

	∇F = parametric_mcp.jacobian_z!.result_buffer
	F = zeros(length(F))
	δz = zeros(length(variables))
	z = zeros(length(variables)) # TODO: add initial guess z₀ using @something

	linsolve = init(LinearProblem(∇F, δz), linear_solve_algorithm)

	#TODO: Add line search for non-LQ case
	# Main Solver loop
	status = :solved
	iters = 0
	max_iters = 50
	tol = 1e-6
	# TODO: make these parameters/input

	kkt_error = Inf
	while kkt_error > tol && iters <= max_iters
		iters += 1

		parametric_mcp.f!(F, z, [parameter_value])
		parametric_mcp.jacobian_z!(∇F, z, [parameter_value])
		linsolve.A = ∇F
		linsolve.b = -F
		solution = solve!(linsolve)

		if !SciMLBase.successful_retcode(solution) &&
		   (solution.retcode !== SciMLBase.ReturnCode.Default)
			verbose &&
				@warn "Linear solve failed. Exiting prematurely. Return code: $(solution.retcode)"
			status = :failed
			break
		end

		δz .= solution.u

		# α = fraction_to_the_boundary_linesearch(z, δz; τ=0.995, decay=0.5, tol=1e-4)
		α_z = 1.0 # TODO: add line search?

		@. z += α_z * δz

		kkt_error = norm(F, Inf)
		verbose && @show norm(F, Inf)
	end

	verbose && @show iters

	return z, status
end

"""       
∇F(z; ϵ) δz = -F(z; ϵ).
"""
# TODO: Rewrite so that it takes conditions and solves directly instead of iteratively.
# TODO: Do initialization to create mcp_obj ahead of time and then add an argument here.
function solve_with_linsolve_too_big(πs, variables, θ, parameter_value; linear_solve_algorithm = LinearSolve.UMFPACKFactorization(), to=TimerOutput(), verbose = false)
	symbolic_type = eltype(variables)

	@timeit to "[Linear Solve] Setup ParametricMCP Inputs" begin
		# Final MCP vector: leader stationarity + leader constraints + follower KKT
		F = Vector{symbolic_type}([
			vcat(collect(values(πs))...)..., # KKT conditions of all players
		])
		F̃ = vcat(F, zeros(length(variables) - length(F)))

		z̲ = fill(-Inf, length(F̃));
		z̅ = fill(Inf, length(F̃))

		# Form mcp via PATH
		println(length(variables), " variables, ", length(F), " conditions")
		mcp_obj = ParametricMCPs.ParametricMCP(F̃, variables, [θ], z̲, z̅; compute_sensitivities = false)
	end
	@timeit to "[Linear Solve] ParametricMCP Setup" begin
		∇F̃ = mcp_obj.jacobian_z!.result_buffer
		∇F = ∇F̃[1:length(F), 1:length(F)]
		println("∇F size: ", size(∇F))
		F = zeros(length(F))
		δz = zeros(length(F))
		δz̃ = zeros(length(F̃))
		println("F size: ", size(F))
		println("δz size: ", size(δz))

	end

	@timeit to "[Linear Solve] Solver Setup" begin
		# TODO: Move outside so we do it once.
		linsolve = init(LinearProblem(∇F, δz), linear_solve_algorithm)

		#TODO: Add line search for non-LQ case
		status = :solved

		mcp_obj.f!(F̃, δz̃, [parameter_value]) # TODO: z instead of δz
		mcp_obj.jacobian_z!(∇F̃, δz̃, [parameter_value])
		linsolve.A = ∇F
		linsolve.b = -F
	end

	@timeit to "[Linear Solve] Call to Solver" begin
		solution = solve!(linsolve)
	end

	if !SciMLBase.successful_retcode(solution) &&
	   (solution.retcode !== SciMLBase.ReturnCode.Default)
		verbose &&
			@warn "Linear solve failed. Exiting prematurely. Return code: $(solution.retcode)"
		status = :failed
		# break
	else
		z_sol = solution.u
	end

	# end for

	return z_sol, status
end


"""       
∇F(z; ϵ) δz = -F(z; ϵ).
"""
# TODO: Rewrite so that it takes conditions and solves directly instead of iteratively.
# TODO: Do initialization to create mcp_obj ahead of time and then add an argument here.
function unoptimized_solve_with_linsolve(πs, variables, vectorized_Ks, K_evals_vec; linear_solve_algorithm = LinearSolve.UMFPACKFactorization(), to=TimerOutput(), verbose = false)
	symbolic_type = eltype(variables)

	@timeit to "[Linear Solve] Setup ParametricMCP Inputs" begin
		# Final MCP vector: leader stationarity + leader constraints + follower KKT
		F = Vector{symbolic_type}([
			vcat(collect(values(πs))...)..., # KKT conditions of all players
		])

		z̲ = fill(-Inf, length(F));
		z̅ = fill(Inf, length(F))

		# Form mcp via ParametricMCP initialization.
		mcp_obj = ParametricMCPs.ParametricMCP(F, variables, vectorized_Ks, z̲, z̅; compute_sensitivities = false)
	end
	@timeit to "[Linear Solve] ParametricMCP Setup" begin
		∇F = mcp_obj.jacobian_z!.result_buffer
		F = zeros(length(F))
		δz = zeros(length(F))
	end

	@timeit to "[Linear Solve] Solver Setup" begin
		# TODO: Move outside so we do it once.
		linsolve = init(LinearProblem(∇F, δz), linear_solve_algorithm)
	end

	@timeit to "[Linear Solve] Jacobian Eval + Problem Definition" begin

		#TODO: Add line search for non-LQ case
		status = :solved

		mcp_obj.f!(F, δz, K_evals_vec) # TODO: z instead of δz
		mcp_obj.jacobian_z!(∇F, δz, K_evals_vec)
		linsolve.A = ∇F
		linsolve.b = -F
	end

	@timeit to "[Linear Solve] Call to Solver" begin
		solution = solve!(linsolve)
	end

	if !SciMLBase.successful_retcode(solution) &&
	   (solution.retcode !== SciMLBase.ReturnCode.Default)
		verbose &&
			@warn "Linear solve failed. Exiting prematurely. Return code: $(solution.retcode)"
		status = :failed
		# break
	else
		z_sol = solution.u
	end

	# end for

	return z_sol, status
end


function solve_with_linsolve(mcp_obj, K_evals_vec; linear_solve_algorithm = LinearSolve.UMFPACKFactorization(), to=TimerOutput(), verbose = false)
	
	@timeit to "[Linear Solve] ParametricMCP Setup" begin
		∇F = mcp_obj.jacobian_z!.result_buffer

		F_size = size(∇F, 1)
		F = zeros(F_size)
		δz = zeros(F_size)
	end

	@timeit to "[Linear Solve] Solver Setup" begin
		# TODO: Move outside so we do it once.
		linsolve = init(LinearProblem(∇F, δz), linear_solve_algorithm)
	end

	@timeit to "[Linear Solve] Jacobian Eval + Problem Definition" begin

		#TODO: Add line search for non-LQ case
		linsolve_status = :solved

		# Main.@infiltrate
		mcp_obj.f!(F, δz, K_evals_vec) # TODO: z instead of δz
		mcp_obj.jacobian_z!(∇F, δz, K_evals_vec)
		linsolve.A = ∇F
		linsolve.b = -F
	end

	@timeit to "[Linear Solve] Call to Solver" begin
		solution = solve!(linsolve)
	end

	if !SciMLBase.successful_retcode(solution) &&
	   (solution.retcode !== SciMLBase.ReturnCode.Default)
		verbose &&
			@warn "Linear solve failed. Exiting prematurely. Return code: $(solution.retcode)"
		linsolve_status = :failed
		# break
	else
		z_sol = solution.u
	end

	# end for

	return z_sol, linsolve_status
end


function compare_lq_solvers(H, graph, primal_dimension_per_player, Js, gs; parameter_value = 1e-5, verbose = false)

	# Run the PATH solver through the run_solver call.
	z_sol_path, status, info, all_variables, (; πs, zs, λs, μs, θ) = run_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs; parameter_value, verbose)
	# z_sol_custom, status = custom_solve(πs, all_variables, θ, parameter_value; verbose)
	z_sol_custom = z_sol_path

	@assert isapprox(z_sol_path, z_sol_custom, atol = 1e-4)
	@show status

	z_sol_custom, status, info, all_variables, (; πs, zs, λs, μs, θ)
end

# TODO: Write helpers that setup the variables ahead of time and once so it's not repeated.
function run_nonlq_solver(H, graph, primal_dimension_per_player, Js, gs, z0_guess=nothing; 
						  parameter_value = 1e-5, max_iters = 3, tol = 1e-6, verbose = false)
	N = nv(graph) # number of players

	#TODO: Add line search for non-LQ case
	# Main Solver loop
	nonlq_solver_status = :solved
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
		# solver_fn = setup_fast_kkt_solver(graph, Js, zs, λs, μs, gs, ws, ys, θ; all_variables, to)
		solver_fn, out_all_augment_variables, setup_info = setup_fast_kkt_solver(graph, Js, zs, λs, μs, gs, ws, ys, θ, all_variables, backend; to=TimerOutput(), verbose = false)
		K_syms = setup_info.K_syms
		πs = setup_info.πs
		# Main.@infiltrate
		all_vectorized_Ks = vcat(map(ii -> reshape(@something(K_syms[ii], Symbolics.Num[]), :), 1:N)...)
		println("all_vectorized_Ks size: ", size(all_vectorized_Ks), eltype(all_vectorized_Ks))
		π_sizes = setup_info.π_sizes
	end

	symbolic_type = eltype(all_variables)

	@timeit to "[Linear Solve] Setup ParametricMCP" begin
		# Final MCP vector: leader stationarity + leader constraints + follower KKT
		F = Vector{symbolic_type}([
			vcat(collect(values(πs))...)..., # KKT conditions of all players
		])

		z̲ = fill(-Inf, length(F));
		z̅ = fill(Inf, length(F))

		# Form mcp via ParametricMCP initialization.
		println(length(all_variables), " variables, ", length(F), " conditions")
		mcp_obj = ParametricMCPs.ParametricMCP(F, all_variables, all_vectorized_Ks, z̲, z̅; compute_sensitivities = false)
	end

	# Main.@infiltrate

	# Initial guess for primal and dual variables.
	z_est = @something(z0_guess, zeros(length(all_variables)))
	πs_est = nothing

	convergence_criterion = Inf
	while convergence_criterion > tol && iters <= max_iters
		@timeit to "Iterative Loop" begin
			# println("Iteration $iters $(z_est[1:5])")
			iters += 1

			@timeit to "Iterative Loop[KKT Conditions with z0]" begin
				# πs, _, _ = get_lq_kkt_conditions(graph, Js, zs, λs, μs, gs, ws, ys, θ; all_variables, z_est, to)
				# πs, _, _ = get_lq_kkt_conditions_optimized(graph, Js, zs, λs, μs, gs, ws, ys, θ; all_variables, z_est, to)
				est_πs, _, _, info = solver_fn(z_est)
				println(length.(collect(values(est_πs))), " KKT conditions")

				# Main.@infiltrate

				# Compute the numeric vectors (of type Float64, generally).
				K_evals = info.K_evals
				for ii in 1:N
					K_evals[ii] = @something(K_evals[ii], Float64[])
					# println(ii, eltype(K_evals[ii]))
					# println("K_evals[$ii] size: ", K_evals[ii])
				end
				# println("Variables eltype: ", eltype(all_variables))
				# for ii in 1:N
				# 	println("K_evals[$ii] eltype: ", eltype(K_evals[ii]))
				# end
				all_vectorized_Kevals = vcat(map(ii -> reshape(@something(K_evals[ii], Float64[]), :), 1:N)...)
				# augmented_z = vcat(z_est, all_vectorized_Kevals)
			end

			@timeit to "Iterative Loop[Linear Solve]" begin
				# dz_sol, linsolve_status = unoptimized_solve_with_linsolve(est_πs, all_variables, [θ], parameter_value; to)
				dz_sol, linsolve_status = solve_with_linsolve(mcp_obj, all_vectorized_Kevals; to)
			end
			# println("dz_sol: ", dz_sol[1:5])

			if linsolve_status != :solved
				@warn "Linear solve failed. Exiting prematurely. Return code: $(linsolve_status)"
				break
			end

			# Update the estimate.
			# TODO: Using a constant step size for now. Add line search here if we see oscillation.
			α = 1.
			next_z_est = z_est .+ α * dz_sol

			# Update the convergence criterion.
			# TODO: Switch to gradient norm.
			convergence_criterion = norm(next_z_est - z_est, Inf)

			# Update the current estimate.
			z_est = next_z_est
			πs_est = πs
		end
		# println("Loop time: ", loop_time)
	end

	show(to)
	println()

	# TODO: Redo to add more general output.
	info = Dict()
	πs = πs_est
	z_est, nonlq_solver_status, info, all_variables, (; πs, zs, λs, μs, θ)
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
	N = 3 # number of players

	# Set up the information structure.
	# This defines a stackelberg chain with three players, where P1 is the leader of P2, and P1+P2 are leaders of P3.
	G = SimpleDiGraph(N);
	add_edge!(G, 2, 1); # P2 -> P1
	add_edge!(G, 2, 3); # P2 -> P3


	H = 1
	Hp1 = H+1 # number of planning stages is 1 for OL game.

	# Helper function
	flatten(vs) = collect(Iterators.flatten(vs))

	# Initial sizing of various dimensions.
	N = 3 # number of players
	T = 3 # time horizon
	state_dimension = 2 # player 1,2,3 state dimension
	control_dimension = 2 # player 1,2,3 control dimension
	x_dim = state_dimension * (T+1)
	u_dim = control_dimension * (T+1)
	aggre_state_dimension = x_dim * N
	aggre_control_dimension = u_dim * N
	total_dimension = aggre_state_dimension + aggre_control_dimension
	primal_dimension_per_player = x_dim + u_dim


	#### player objectives ####
	# player 3 (follower)'s objective function: P3 follows P2
	function J₃(z₁, z₂, z₃, θ)
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
		xs³, us³ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
		xs², us² = xs, us
		0.5*sum((xs³[end] .- xs²[end]) .^ 2) + 0.05*sum(sum(u³ .^ 2) for u³ in us³) + 0.05*sum(sum(u² .^ 2) for u² in us²)
	end

	# player 2 (leader)'s objective function: P2 wants P1 and P3 to get to the origin
	function J₂(z₁, z₂, z₃, θ)
		(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
		xs³, us³ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
		xs², us² = xs, us
		(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
		xs¹, us¹ = xs, us
		sum((0.5*(xs¹[end] .+ xs³[end])) .^ 2) + 0.05*sum(sum(u .^ 2) for u in us²)
	end

	# player 1 (top leader)'s objective function: P1 wants to get close to P2's final position
	function J₁(z₁, z₂, z₃, θ)
		(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
		xs¹, us¹ = xs, us
		(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
		xs², us² = xs, us
		0.5*sum((xs¹[end] .- xs²[end]) .^ 2) + 0.05*sum(sum(u .^ 2) for u in us¹)
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

	dynamics_constraint(zᵢ) =
		mapreduce(vcat, 1:T) do t
			dynamics(zᵢ, t)
		end

	gs = [function (zᵢ)
		  vcat(dynamics_constraint(zᵢ),
			   make_ic_constraint(i)(zᵢ))
	end for i in 1:N] # each player has the same dynamics constraint

	parameter_value = 1e-5
	# Main.@infiltrate
	z_sol, status, info, all_variables, vars = compare_lq_solvers(H, G, primal_dimension_per_player, Js, gs; parameter_value, verbose)
	(; πs, zs, λs, μs, θ) = vars


	z₁ = zs[1]
	z₂ = zs[2]
	z₃ = zs[3]
	z₁_sol = z_sol[1:length(z₁)]
	z₂_sol = z_sol[(length(z₁)+1):(length(z₁)+length(z₂))]
	z₃_sol = z_sol[(length(z₁)+length(z₂)+1):(length(z₁)+length(z₂)+length(z₃))]


	# Evaluate the KKT residuals at the solution to check solution quality.
	z_sols = [z₁_sol, z₂_sol, z₃_sol]
	# Main.@infiltrate
	# evaluate_kkt_residuals(πs, all_variables, z_sol, θ, parameter_value; verbose = verbose)

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
