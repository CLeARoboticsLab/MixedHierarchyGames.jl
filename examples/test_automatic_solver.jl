
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

include("graph_utils.jl")
include("make_symbolic_variables.jl")
include("solve_kkt_conditions.jl")
include("evaluate_results.jl")
include("general_kkt_construction.jl")



"""                      
∇F(z; ϵ) δz = -F(z; ϵ).
"""
# TODO: Rewrite so that it takes conditions and solves directly instead of iteratively.
function custom_solve(πs, variables, θ, parameter_value; linear_solve_algorithm = LinearSolve.UMFPACKFactorization(), verbose = false)
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


function run_solver(H, graph, primal_dimension_per_player, Js, gs; parameter_value = 1e-5, verbose = false)
	N = nv(graph) # number of players

	# Construct symbols for each player's decision variables.
	# TODO: Construct sizes and orderings.
	backend = SymbolicTracingUtils.SymbolicsBackend()
	zs = [SymbolicTracingUtils.make_variables(
		backend,
		make_symbolic_variable(:z, i, H),
		primal_dimension_per_player,
	) for i in 1:N]

	λs = [SymbolicTracingUtils.make_variables(
		backend,
		make_symbolic_variable(:λ, i, H),
		length(gs[i](zs[i])),
	) for i in 1:N]

	μs = Dict{Tuple{Int, Int}, Any}()
	ws = Dict{Int, Any}()
	ys = Dict{Int, Any}()
	for i in 1:N
		# TODO: Replace this call with a built-in search ordering.
		# yᵢ is the information vector containing states zᴸ associated with leaders L of i.
		# This call must ensure the highest leader comes first in the ordering.
		leaders = get_all_leaders(graph, i)
		ys[i] = vcat(zs[leaders]...)
		ws[i] = zs[i] # Initialize vector with current agent's state.

		# wᵢ is used to identify policy constraints by leaders of i.
		# Construct wᵢ by adding (1) zs which are not from leaders of i and not i itself,
		for jj in 1:N
			if jj in leaders || jj == i
				continue
			end
			ws[i] = vcat(ws[i], zs[jj])
		end

		#                        (2) λs of i and its followers, and
		for jj in BFSIterator(graph, i)
			ws[i] = vcat(ws[i], λs[jj])
		end

		#                        (3) μs associated with i's follower policies.
		# TODO: Replace this call with a built-in search ordering (mix with BFS above).
		# Get all followers of i, create the variable for each, and store them in a Dict.
		followers = get_all_followers(graph, i)
		for j in followers
			μs[(i, j)] = SymbolicTracingUtils.make_variables(
				backend,
				make_symbolic_variable(:μ, i, j, H),
				primal_dimension_per_player,
			)
			ws[i] = vcat(ws[i], μs[(i, j)])
		end


		if verbose
			println("ws for P$i ($(length(ws[i]))):\n", ws[i])
			println()
			println("ys for P$i ($(length(ys[i]))):\n", ys[i])
			println()
		end
	end
	θ = only(SymbolicTracingUtils.make_variables(backend, :θ, 1))
	Main.@infiltrate

	πs, _, _ = get_lq_kkt_conditions(graph, Js, zs, λs, μs, gs, ws, ys, θ)

	# Construct a list of all variables in order and solve.
	temp = vcat(collect(values(μs))...)
	all_variables = vcat(vcat(zs...), vcat(λs...))
	if !isempty(temp)
		all_variables = vcat(all_variables, vcat(collect(values(μs))...))
	end

	z_sol, status, info = solve_with_path(πs, all_variables, θ, parameter_value)

	z_sol_custom, status = custom_solve(πs, all_variables, θ, parameter_value; verbose)

	@assert isapprox(z_sol, z_sol_custom, atol = 1e-4)
	@show status
	z_sol_custom, status, info, all_variables, (; πs, zs, λs, μs, θ)
end

######### INPUT: Initial conditions ##########################
x0 = [
	[0.0; 2.0], # [px, py]
	[2.0; 4.0],
	[6.0; 8.0],
]
###############################################################

# Main body of algorithm implementation. Will restructure as needed.
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
	Main.@infiltrate
	z_sol, status, info, all_variables, vars = run_solver(H, G, primal_dimension_per_player, Js, gs; parameter_value, verbose)
	(; πs, zs, λs, μs, θ) = vars


	z₁ = zs[1]
	z₂ = zs[2]
	z₃ = zs[3]
	z₁_sol = z_sol[1:length(z₁)]
	z₂_sol = z_sol[(length(z₁)+1):(length(z₁)+length(z₂))]
	z₃_sol = z_sol[(length(z₁)+length(z₂)+1):(length(z₁)+length(z₂)+length(z₃))]


	# Evaluate the KKT residuals at the solution to check solution quality.
	z_sols = [z₁_sol, z₂_sol, z₃_sol]
	evaluate_kkt_residuals(πs, all_variables, z_sol, θ, parameter_value; verbose = verbose)

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
