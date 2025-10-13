
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes
using Graphs
using InvertedIndices
using LinearAlgebra: I, norm, pinv, Diagonal, rank
using Plots
using Symbolics
using SymbolicTracingUtils
using TrajectoryGamesBase: unflatten_trajectory

include("graph_utils.jl")
include("make_symbolic_variables.jl")
include("solve_kkt_conditions.jl")
include("evaluate_results.jl")
include("general_kkt_construction.jl")


function setup_problem_variables(H, graph, primal_dimension_per_player, gs; verbose = false)
	N = nv(graph) # number of players

	# Construct symbols for each player's decision variables.
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

	# Construct a list of all variables in order and solve.
	temp = vcat(collect(values(μs))...)
	all_variables = vcat(vcat(zs...), vcat(λs...))
	if !isempty(temp)
		all_variables = vcat(all_variables, vcat(collect(values(μs))...))
	end

	(; all_variables, zs, λs, μs, θ, ws, ys)
end

function run_solver(H, graph, primal_dimension_per_player, Js, gs; parameter_value = 1e-5, verbose = false)
	N = nv(graph) # number of players

	# Construct symbols for each player's decision variables.
	(; all_variables, zs, λs, μs, θ, ws, ys) = setup_problem_variables(H, graph, primal_dimension_per_player, gs; verbose)

	πs, _, _ = get_lq_kkt_conditions(graph, Js, zs, λs, μs, gs, ws, ys, θ)

	# Construct a list of all variables in order and solve.
	temp = vcat(collect(values(μs))...)
	all_variables = vcat(vcat(zs...), vcat(λs...))
	if !isempty(temp)
		all_variables = vcat(all_variables, vcat(collect(values(μs))...))
	end
	z_sol, status, info = solve_with_path(πs, all_variables, θ, parameter_value)

	z_sol, status, info, all_variables, (; πs, zs, λs, μs, θ)
end

# Main body of algorithm implementation. Will restructure as needed.
function main(verbose = false)
	N = 3 # number of players

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
	N = 3 # number of players
	T = 10 # time horizon
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
	# A = I(state_dimension * num_players)
	# B¹ = [Δt * I(control_dimension); zeros(4, 2)]
	# B² = [zeros(2, 2); Δt * I(control_dimension); zeros(2, 2)]
	# B³ = [zeros(4, 2); Δt * I(control_dimension)]
	# B = [B¹ B² B³]

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
	ics = [[0.0; 2.0],
	       [2.0; 4.0],
		   [6.0; 8.0]] # initial conditions for each player

	make_ic_constraint(i) = function (zᵢ)
		(; xs, us) = unflatten_trajectory(zᵢ, state_dimension, control_dimension)
		x1 = xs[1]
		return x1 - ics[i]
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
	z_sol, status, info, all_variables, vars = run_solver(H, G, primal_dimension_per_player, Js, gs; parameter_value, verbose = verbose)
	(; πs, zs, λs, μs, θ) = vars


	# Print some information about the solution.
	println("z-lengths: ", length(zs[1]), " ", length(zs[2]), " ", length(zs[3]))
	println("λ-lengths: ", length(λs[1]), " ", length(λs[2]), " ", length(λs[3]))
	println("μ-lengths: ", [length(μs[(i, j)]) for (i, j) in keys(μs)])
	println("Solution status: ", size(z_sol), " ", z_sol[1:36])


	z₁ = zs[1]
	z₂ = zs[2]
	z₃ = zs[3]
	z₁_sol = z_sol[1:length(z₁)]
	z₂_sol = z_sol[(length(z₁)+1):(length(z₁)+length(z₂))]
	z₃_sol = z_sol[(length(z₁)+length(z₂)+1):(length(z₁)+length(z₂)+length(z₃))]


	# Evaluate the KKT residuals at the solution to check solution quality.

	z_sols = [z₁_sol, z₂_sol, z₃_sol]
	evaluate_kkt_residuals(πs, all_variables, z_sol, θ, parameter_value; verbose = verbose)

	(; xs, us) = unflatten_trajectory(z₁_sol, state_dimension, control_dimension)
	println("P1 (x,u) solution : ($xs, $us)")
	println("P1 Objective: $(Js[1](z₁_sol, z₂_sol, z₃_sol, 0))")
	(; xs, us) = unflatten_trajectory(z₂_sol, state_dimension, control_dimension)
	println("P2 (x,u) solution : ($xs, $us)")
	println("P2 Objective: $(Js[2](z₁_sol, z₂_sol, z₃_sol, 0))")
	(; xs, us) = unflatten_trajectory(z₃_sol, state_dimension, control_dimension)
	println("P3 (x,u) solution : ($xs, $us)")
	println("P3 Objective: $(Js[3](z₁_sol, z₂_sol, z₃_sol, 0))")

	# Plot the trajectories of each player.
	# Helper: turn the vector-of-vectors `xs` into a 2×(T+1) matrix
	state_matrix(xs_vec) = hcat(xs_vec...)  # each column is x at time t

	# Reconstruct trajectories from solutions
	xs1, _ = unflatten_trajectory(z₁_sol, state_dimension, control_dimension)
	xs2, _ = unflatten_trajectory(z₂_sol, state_dimension, control_dimension)
	xs3, _ = unflatten_trajectory(z₃_sol, state_dimension, control_dimension)

	X1 = state_matrix(xs1)  # 2 × (T+1)
	X2 = state_matrix(xs2)
	X3 = state_matrix(xs3)

	# Plot 2D paths
	plt = plot(; xlabel = "x₁", ylabel = "x₂", title = "Player Trajectories (T=$(T), Δt=$(Δt))",
		legend = :bottomright, aspect_ratio = :equal, grid = true)

	plot!(plt, X1[1, :], X1[2, :]; lw = 2, marker = :circle, ms = 3, label = "P1")
	plot!(plt, X2[1, :], X2[2, :]; lw = 2, marker = :diamond, ms = 4, label = "P2")
	plot!(plt, X3[1, :], X3[2, :]; lw = 2, marker = :utriangle, ms = 4, label = "P3")

	# Mark start (t=0) and end (t=T) points
	scatter!(plt, [X1[1, 1], X2[1, 1], X3[1, 1]], [X1[2, 1], X2[2, 1], X3[2, 1]];
		markershape = :star5, ms = 8, label = "start (t=0)")
	scatter!(plt, [X1[1, end], X2[1, end], X3[1, end]], [X1[2, end], X2[2, end], X3[2, end]];
		markershape = :hexagon, ms = 8, label = "end (t=$T)")


	# Origin
	scatter!(plt, [0.0], [0.0]; marker = :cross, ms = 8, color = :black, label = "Origin (0,0)")

	display(plt)

	return z_sol, status, info, πs, all_variables
end
