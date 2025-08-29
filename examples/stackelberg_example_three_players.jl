
using ParametricMCPs: ParametricMCPs

using LinearAlgebra: I, norm, pinv, Diagonal, rank

using Symbolics
using SymbolicTracingUtils
using BlockArrays: BlockArrays, BlockArray, Block, blocks, blocksizes
using TrajectoryGamesBase: unflatten_trajectory
using InvertedIndices

using Plots

# Helper function
flatten(vs) = collect(Iterators.flatten(vs))

num_players = 3 # number of players
T = 10 # time horizon
state_dimension = 2 # player 1,2,3 state dimension
control_dimension = 2 # player 1,2,3 control dimension
x_dim = state_dimension * (T+1)
u_dim = control_dimension * (T+1)
aggre_state_dimension = x_dim * num_players
aggre_control_dimension = u_dim * num_players
total_dimension = aggre_state_dimension + aggre_control_dimension
primal_dimension_per_player = x_dim + u_dim

# Dynamics
Δt = 0.5 # time step
A = I(state_dimension * num_players)
B¹ = [Δt * I(control_dimension); zeros(4, 2)]
B² = [zeros(2, 2); Δt * I(control_dimension); zeros(2, 2)]
B³ = [zeros(4, 2); Δt * I(control_dimension)]
B = [B¹ B² B³]


##### Solution via PATH ######

# player 3 (follower)'s objective function: P3 follows P2
function J₃(z₃, z₂, θ)
	(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
	xs³, us³ = xs, us
	(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
	xs², us² = xs, us
	0.5*sum((xs³[end] .- xs²[end]) .^ 2) + 0.05*sum(sum(u³ .^ 2) for u³ in us³) #+ 0.05*sum(sum(u² .^ 2) for u² in us²)
end

# player 2 (middle leader)'s objective function: P2 follows P1
function J₂(z₂, z₁, z₃, θ)
	# (; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
	# xs³, us³ = xs, us
	(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
	xs², us² = xs, us
	(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
	xs¹, us¹ = xs, us
	# sum((0.5*(xs¹[end] .+ xs³[end])) .^ 2) + 0.05*sum(sum(u .^ 2) for u in us²)
	0.5*sum((xs²[end] .- xs¹[end]) .^ 2) + 0.05*sum(sum(u² .^ 2) for u² in us²)
end

# player 1 (leader)'s objective function: P1 wants to get to origin
function J₁(z₁, z₂, θ)
	(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
	xs¹, us¹ = xs, us
	0.5*sum(xs¹[end] .^ 2) + 0.05*sum(sum(u .^ 2) for u in us¹)
end

function dynamics(z, t)
	(; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
	x = xs[t]
	u = us[t]
	xp1 = xs[t+1]
	return xp1 - x - Δt*u
end

g₁(z₁) =
	mapreduce(vcat, 1:T) do t
		dynamics(z₁, t)
	end
g₂(z₂) =
	mapreduce(vcat, 1:T) do t
		dynamics(z₂, t)
	end
g₃(z₃) =
	mapreduce(vcat, 1:T) do t
		dynamics(z₃, t)
	end

# Symbolic variables
backend = SymbolicTracingUtils.SymbolicsBackend()
z₁ = SymbolicTracingUtils.make_variables(
	backend,
	Symbol("z̃1"),
	primal_dimension_per_player,
)
z₂ = SymbolicTracingUtils.make_variables(
	backend,
	Symbol("z̃2"),
	primal_dimension_per_player,
)
z₃ = SymbolicTracingUtils.make_variables(
	backend,
	Symbol("z̃3"),
	primal_dimension_per_player,
)
symbolic_type = eltype(z₂)
θ = only(SymbolicTracingUtils.make_variables(backend, :θ, 1))
(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
xs¹, us¹ = xs, us
(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
xs², us² = xs, us
(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
xs³, us³ = xs, us

# Initial conditions
ic₁ = xs¹[1] .- [0.0; 2.0]
ic₂ = xs²[1] .- [2.0; 4.0]
ic₃ = xs³[1] .- [6.0; 8.0]

# KKT conditions of p3 (follower)
λ₃ = SymbolicTracingUtils.make_variables(backend, Symbol("λ_3"), length(g₃(z₃)) + length(ic₃))
L₃ = J₃(z₃, z₂, θ) - λ₃' * [g₃(z₃); ic₃]
π₃ = vcat(
	Symbolics.gradient(L₃, z₃), 
	g₃(z₃), 
	ic₃
)  # stationarity of follower only w.r.t its own vars


# KKT conditions of p2 (middle-leader)
p₂_equality_dim = length(g₂(z₂)) + length(ic₂)
p₂_lower_level_stationarity_dim = length(z₃)
λ₂₁ = SymbolicTracingUtils.make_variables(backend, Symbol("λ_2_1"), p₂_equality_dim)
λ₂₂ = SymbolicTracingUtils.make_variables(backend, Symbol("λ_2_2"), p₂_lower_level_stationarity_dim)
λ₂ = Vector{Symbolics.Num}() # ASK
push!(λ₂, λ₂₁..., λ₂₂...)


M₂, N₂ = let
	M = Symbolics.jacobian(π₃, [z₃; λ₃])
	N = Symbolics.jacobian(π₃, [z₁; z₂])
	M, N
end
ϕ³ = - hcat(I(length(z₃)), zeros(length(z₃), length(λ₃))) * (M₂ \ N₂ * [z₁; z₂])

L² = J₂(z₂, z₁, z₃, θ) - λ₂₁' * [g₂(z₂); ic₂] - λ₂₂' * (z₃ .- ϕ³)
# L² = J₂(z₂, z₁, z₃, θ) - λ₂' * [g₂(z₂); ic₂; z₃ .- ϕ³]
π₂ = vcat(
	Symbolics.gradient(L², z₂),
	Symbolics.gradient(L², z₃),
	g₂(z₂),
	ic₂,
)

# KKT conditions of p1 (leader)
p₁_equality_dim = length(g₁(z₁)) + length(ic₁)
p₁_lower_level_stationarity_dim₂ = length(z₂) 
p₁_lower_level_stationarity_dim₃ = length(z₃)
λ₁₁ = SymbolicTracingUtils.make_variables(backend, Symbol("λ_1_1"), p₁_equality_dim)
λ₁₂ = SymbolicTracingUtils.make_variables(backend, Symbol("λ_1_2"), p₁_lower_level_stationarity_dim₂)
λ₁₃ = SymbolicTracingUtils.make_variables(backend, Symbol("λ_1_3"), p₁_lower_level_stationarity_dim₃)
λ₁	 = Vector{Symbolics.Num}()
push!(λ₁, λ₁₁..., λ₁₂..., λ₁₃...)

M₁, N₁ = let
	M = Symbolics.jacobian(π₂, [z₂; z₃; λ₂; λ₃])
	N = Symbolics.jacobian(π₂, z₁)
	M, N
end

# M₁ = Symbolics.value.(M₁) if M₁ singular and use pinv(M₁)
# N₁ = Symbolics.value.(N₁)
ϕ² = - hcat(I(length(z₂)), zeros(length(z₂), length(z₃)+length(λ₂)+length(λ₃))) * (M₁ \ N₁ * z₁)

L¹ = J₁(z₁, z₂, θ) - λ₁₁' * [g₁(z₁); ic₁] - λ₁₂' * (z₂ .- ϕ²) - λ₁₃' * (z₃ .- ϕ³)
π₁ = vcat(
	Symbolics.gradient(L¹, z₁),
	Symbolics.gradient(L¹, z₂),
	Symbolics.gradient(L¹, z₃),
	g₁(z₁),
	ic₁,
)

# # KKT conditions of p4 (nash) # TODO
# λ₄ = SymbolicTracingUtils.make_variables(
# 	backend, Symbol("λ_4"), length(g₄(z₄)) + length(ic₄),
# )
# L⁴ = J₄(z₄, θ) - λ₄' * [g₄(z₄); ic₄]
# ∇L₄ = Symbolics.gradient(L₄, z₄)


# Final MCP vector: leader stationarity + leader constraints + follower KKT
F = Vector{symbolic_type}([
	π₁;
	π₂;
	π₃
])

variables = vcat(z₁, z₂, z₃, λ₁, λ₂, λ₃)
z̲ = fill(-Inf, length(F));
z̅ = fill(Inf, length(F))

# Solve via PATH
parameter_value = [1e-5]
parametric_mcp = ParametricMCPs.ParametricMCP(F, variables, [θ], z̲, z̅; compute_sensitivities = false)
z_sol, status, info = ParametricMCPs.solve(
	parametric_mcp,
	parameter_value;
	initial_guess = zeros(length(variables)),
	verbose = false,
	cumulative_iteration_limit = 100000,
	proximal_perturbation = 1e-2,
	# major_iteration_limit = 1000,
	# minor_iteration_limit = 2000,
	# nms_initial_reference_factor = 50,
	use_basics = true,
	use_start = true,
)
@show status
z₁_sol = z_sol[1:length(z₁)]
z₂_sol = z_sol[(length(z₁)+1):(length(z₁)+length(z₂))]
z₃_sol = z_sol[(length(z₁)+length(z₂)+1):(length(z₁)+length(z₂)+length(z₃))]
(; xs, us) = unflatten_trajectory(z₁_sol, state_dimension, control_dimension)
println("P1 (x,u) solution : ($xs, $us)")
println("P1 Objective: $(J₁(z₁_sol, z₂_sol, 0))")
(; xs, us) = unflatten_trajectory(z₂_sol, state_dimension, control_dimension)
println("P2 (x,u) solution : ($xs, $us)")
println("P2 Objective: $(J₂(z₂_sol, z₁_sol, z₃_sol, 0))")
(; xs, us) = unflatten_trajectory(z₃_sol, state_dimension, control_dimension)
println("P3 (x,u) solution : ($xs, $us)")
println("P3 Objective: $(J₃(z₃_sol, z₂_sol, 0))")



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
