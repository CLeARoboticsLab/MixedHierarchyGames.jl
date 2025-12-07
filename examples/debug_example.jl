using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes
using Graphs
using InvertedIndices
using LinearAlgebra: I, norm, pinv, Diagonal, rank
using LinearSolve: LinearSolve, LinearProblem, init, solve!
using Logging
using Plots
using SciMLBase: SciMLBase
using Symbolics
using SymbolicTracingUtils
using TimerOutputs
using TrajectoryGamesBase: unflatten_trajectory

include("graph_utils.jl")
include("make_symbolic_variables.jl")
include("solve_kkt_conditions.jl")
include("evaluate_results.jl")
include("general_kkt_construction.jl")
include("test_automatic_solver.jl")


function setup_problem_variables(H, graph, primal_dimension_per_player, gs; backend = SymbolicTracingUtils.SymbolicsBackend(), verbose = false)
	"""
	Constructs the symbolic variables needed for the problem based on the information structure graph.

	Parameters
	----------
	H (Int) : The number of planning stages (e.g., 1 for open-loop, T for more).
	graph (SimpleDiGraph) : The information structure of the game at each stage, defined as a directed graph.
	primal_dimension_per_player (Vector{Int}) : The dimension of each player's decision variable.
	gs (Vector{Function}) : A vector of equality constraint functions for each player, accepting only that
						    player's decision variable zᵢ.
	backend (SymbolicTracingUtils.Backend) : The symbolic backend to use (default: SymbolicsBackend()).
	verbose (Bool) : Whether to print verbose output (default: false).

	Returns
	-------
	all_variables (Vector{Num}) : A vector of all symbolic variables used in the problem.
	zs (Vector{Vector{Num}}) : A vector of each player's decision variable symbols.
	λs (Vector{Vector{Num}}) : A vector of each player's Lagrange multiplier symbols for their constraints.
	μs (Dict{Tuple{Int, Int}, Vector{Num}}) : A dictionary of Lagrange multiplier symbols for each leader-follower pair.
	θ (Num) : The parameter symbol.
	ys (Dict{Int, Vector{Num}}) : A dictionary of each player's information variable symbols (decision input).
	ws (Dict{Int, Vector{Num}}) : A dictionary of each player's remaining variable symbols (decision output).
	"""

	N = nv(graph) # number of players

	# Construct symbols for each player's decision variables.
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
			@debug "ws for P$i" ws=ws[i] len=length(ws[i])
			@debug "ys for P$i" ys=ys[i] len=length(ys[i])
		end
	end
	θ = only(SymbolicTracingUtils.make_variables(backend, :θ, 1))

	# Construct a list of all variables in order and solve.
	temp = vcat(collect(values(μs))...)
	all_variables = vcat(vcat(zs...), vcat(λs...))
	if !isempty(temp)
		all_variables = vcat(all_variables, vcat(collect(values(μs))...))
	end

	(; all_variables, zs, λs, μs, θ, ys, ws)
end


function run_lq_solver(H, graph, primal_dimension_per_player, Js, gs; parameter_value = 1e-5, verbose = false)
	"""
	Solves a linear-quadratic equality-constrained mathematical programming network problem.

	TODO: T stages (feedback information structure) not supported yet
	TODO: Only equality constraints supported right now. Inequalities will be added later.

	Parameters
	----------
	H (Int) : The number of planning stages (e.g., 1 for open-loop, T for more).
	graph (SimpleDiGraph) : The information structure of the game at each stage, defined as a directed graph.
	primal_dimension_per_player (Vector{Int}) : The dimension of each player's decision variable.
	Js (Dict{Int, Function}) : A dictionary mapping player indices to their objective functions 
							   accepting each player's decision variables, z₁, z₂, ..., zₙ, and the parameter θ.
	gs (Vector{Function}) : A vector of equality constraint functions for each player, accepting only that
						    player's decision variable zᵢ.
	parameter_value (Float64) : The value to assign to the parameter θ in the problem (default: 1e-5).
	verbose (Bool) : Whether to print verbose output (default: false).

	Returns
	-------
	z_sol (Vector{Float64}) : The solution vector containing all players' decision variables.
	status (Symbol) : The status of the solver (e.g., :solved, :failed, :unknown).
	info (Any) : Additional information about the solver
	πs (Dict{Int, Vector{Any}}) : The KKT conditions of each player i, contained in a dictionary indexed by player number.
	all_variables (Vector{Num}) : A vector of all symbolic variables used in the problem.
	"""

	# Number of players in the game.
	N = nv(graph)

	# Construct symbols for each player's decision variables.
	(; all_variables, zs, λs, μs, θ, ws, ys) = setup_problem_variables(H, graph, primal_dimension_per_player, gs; verbose)

	πs, _, _, _ = get_lq_kkt_conditions(graph, Js, zs, λs, μs, gs, ws, ys, θ)

	# Construct a list of all variables in order and solve.
	temp = vcat(collect(values(μs))...)
	all_variables = vcat(vcat(zs...), vcat(λs...))
	if !isempty(temp)
		all_variables = vcat(all_variables, vcat(collect(values(μs))...))
	end
	z_sol, status, info = solve_with_path(πs, all_variables, θ, parameter_value)

	z_sol, status, info, all_variables, (; πs, zs, λs, μs, θ)
end


function get_simple_three_player_qp(;verbose = false)
	# Number of players in the game
	N = 3

	# Set up the information structure (no hierarchy for this simple demo).
	G = SimpleDiGraph(N)
    add_edge!(G, 2, 1); # P1 -> P2
	add_edge!(G, 2, 3); # P2 -> P3

	H = 1
	Hp1 = H+1 # number of planning stages is 1 for OL game.

	# Helper function
	flatten(vs) = collect(Iterators.flatten(vs))

	# Initial sizing of various dimensions.
	state_dimension = 2 # player 1,2,3's state dimension
	# control_dimension = 2 # player 1,2,3's control dimension

	# Additional dimension computations.
	x_dim = state_dimension 
	u_dim = 0 #control_dimension * (T+1)
	total_dimension = x_dim * N
	# aggre_control_dimension = u_dim * N
	# total_dimension = aggre_state_dimension + aggre_control_dimension
	primal_dimension_per_player = x_dim #+ u_dim

	problem_dims = (;
		state_dimension,
		x_dim,
		total_dimension,
		primal_dimension_per_player,
	)

	#### Player Objectives ####
	# Player 1's objective function: P1 wants to get close to P2's final position 
	# considering only its own control effort.
	function J₁(x₁, x₂, x₃, θ)
        (x₁[1] - 0.1)^4 + (x₁[2] - 0.1)^4
	end

	# Player 2's objective function: P2 wants P1 and P3 to get to the origin
	function J₂(x₁, x₂, x₃, θ)
        (x₂[1] - 1)^4 + (x₂[2] - 1)^4
	end

	# Player 3's objective function: P3 wants to get close to P2's final position considering its own and P2's control effort.
	function J₃(x₁, x₂, x₃, θ)
        (x₃[1] - 2)^4 + (x₃[2] - 2)^4
	end

	Js = Dict{Int, Any}(
		1 => J₁,
		2 => J₂,
		3 => J₃,
	)

	# # Set up the equality constraints for each player.
	# ics = [[0.0; 0.0],
	# 	   [2.0; 4.0],
	# 	   [6.0; 8.0]] # initial conditions for each player

	# make_ic_constraint(i) = function (zᵢ)
	# 	return zᵢ - ics[i]
	# end

	# No constraints for this simplified example; return a zero-length vector with matching element type.
	gs = [function (zᵢ) zeros(eltype(zᵢ), 0) end for i in 1:N]

	return N, G, H, problem_dims, Js, gs
end

function solve_simple_nonqp(H, graph, primal_dimension_per_player, Js, gs; parameter_value = 1e-5, verbose = false)
	z_sol_path, path_status, info, all_variables, (; πs, zs, λs, μs, θ), (;) = run_nonlq_solver(
		H, graph, primal_dimension_per_player, Js, gs, nothing;
		parameter_value = 1e-5, max_iters = 30, tol = 1e-6, verbose = false,
		ls_α_init=1.0, ls_β=0.5, ls_c₁=1e-4, max_ls_iters=10,
		to = TimerOutput(), backend=SymbolicTracingUtils.SymbolicsBackend(),
		preoptimization_info=nothing,
	)

	return z_sol_path, path_status, info, all_variables, (; πs, zs, λs, μs, θ)
end

function solve_simple_qp(H, graph, primal_dimension_per_player, Js, gs; parameter_value = 1e-5, verbose = false)
	"""
	Compares the solution to an LQ hierarchical game using the PATH solver and using a custom linear solver.
	Ensures that both solvers return the same solution on the LQ problem.
	"""

	# Run the PATH solver through the run_solver call.
	z_sol_path, path_status, info, all_variables, (; πs, zs, λs, μs, θ) = run_lq_solver(H, graph, primal_dimension_per_player, Js, gs; parameter_value, verbose)
	z_sol_linsolve, linsolve_status = lq_game_linsolve(πs, all_variables, θ, parameter_value; verbose)

	@assert isapprox(z_sol_path, z_sol_linsolve, atol = 1e-4)
	verbose && @info "PATH status: $path_status"
	verbose && @info "LinSolve status: $linsolve_status"

	z_sol_path, path_status, z_sol_linsolve, linsolve_status, info, all_variables, (; πs, zs, λs, μs, θ)
end

# Main body of algorithm implementation for simple example. Will restructure as needed.
function main(verbose = false)
	N, G, H, problem_dims, Js, gs = get_simple_three_player_qp(; verbose)

	primal_dimension_per_player = problem_dims.primal_dimension_per_player
	state_dimension = problem_dims.state_dimension

	# Print dimension information.
	@info "Problem dimensions:\n" *
		"  Number of players: $N\n" *
		"  Number of Stages: $H (OL = 1; FB > 1)\n" *
		"  Dimension per player: $(primal_dimension_per_player)\n" *
		"  Total primal dimension: $(problem_dims.total_dimension)"

    ### Solve the LQ game using the automatic solver. ###
    parameter_value = 1e-5
    z_sol, status, info, all_variables, vars = solve_simple_nonqp(H, G, primal_dimension_per_player, Js, gs; parameter_value, verbose)
	(; πs, zs, λs, μs, θ) = vars

	# Extract each player's solution from the full solution vector.
	z₁ = zs[1]
	z₂ = zs[2]
	z₃ = zs[3]
	z₁_sol = z_sol[1:length(z₁)]
	z₂_sol = z_sol[(length(z₁)+1):(length(z₁)+length(z₂))]
	z₃_sol = z_sol[(length(z₁)+length(z₂)+1):(length(z₁)+length(z₂)+length(z₃))]

	# # Evaluate the KKT residuals at the solution to check solution quality.
	# evaluate_kkt_residuals(πs, all_variables, z_sol, θ, parameter_value; verbose = verbose)

	# Print solution information.
	z_sols = [z₁_sol, z₂_sol, z₃_sol]# Report objective value for each agent at the solved trajectories.
	costs = [Js[i](z_sols[1], z_sols[2], z_sols[3], parameter_value) for i in 1:N]
	@info "Agent costs" costs=costs
	verbose && print_solution_info(z_sols, Js, problem_dims)

		# Plot the trajectories of each player.
		plot_player_results(z_sols, problem_dims)

	return z_sol, status, info, πs, all_variables
end

function plot_player_results(z_sols, problem_dims)
	state_dimension = problem_dims.state_dimension
		# Extract final 2D state for each player (z already stores the 2D state here).
		finals = [z[1:state_dimension] for z in z_sols]

	plt = plot(; xlabel = "x₁", ylabel = "x₂", title = "Final Player Positions",
		legend = :bottomright, aspect_ratio = :equal, grid = true)

	scatter!(plt, [finals[1][1]], [finals[1][2]]; markershape = :circle, ms = 8, label = "P1")
	scatter!(plt, [finals[2][1]], [finals[2][2]]; markershape = :diamond, ms = 9, label = "P2")
	scatter!(plt, [finals[3][1]], [finals[3][2]]; markershape = :utriangle, ms = 9, label = "P3")

	# Optional reference origin.
	scatter!(plt, [0.0], [0.0]; marker = :cross, ms = 8, color = :black, label = "Origin (0,0)")

	display(plt)
end

function print_solution_info(z_sols, Js, problem_dims)
	"""
	Prints the solution information for each player, including their trajectories and objective values.
	"""
	@info "P1 state" x=z_sols[1]
	@info "P1 Objective" value=Js[1](z_sols[1], z_sols[2], z_sols[3], 0)

	@info "P2 state" x=z_sols[2]
	@info "P2 Objective" value=Js[2](z_sols[1], z_sols[2], z_sols[3], 0)

	@info "P3 state" x=z_sols[3]
	@info "P3 Objective" value=Js[3](z_sols[1], z_sols[2], z_sols[3], 0)
end
