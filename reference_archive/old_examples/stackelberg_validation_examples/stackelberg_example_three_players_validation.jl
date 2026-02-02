using ParametricMCPs: ParametricMCPs
using LinearAlgebra: I, norm
using Symbolics
using SymbolicTracingUtils
using TrajectoryGamesBase: unflatten_trajectory
using InvertedIndices

using Plots
# ------------------------------------------------------------
# Problem setup
# ------------------------------------------------------------
N                           = 3
T                           = 10
state_dimension             = 2
control_dimension           = 2
x_dim                       = state_dimension * (T + 1)
u_dim                       = control_dimension * (T + 1)
primal_dimension_per_player = x_dim + u_dim

# simple single-integrator dynamics step: x_{t+1} = x_t + Δt * u_t
Δt = 0.2

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
flatten(vs) = collect(Iterators.flatten(vs))

"Make a contiguous split (v -> v[1:n1], v[n1+1:n1+n2], v[...])."
split3(v, n1, n2, n3) = (v[1:n1], v[(n1+1):(n1+n2)], v[(n1+n2+1):(n1+n2+n3)])

"Unpack a trajectory vector z into (xs, us) arrays (length T+1 each)."
unpack(z) = unflatten_trajectory(z, state_dimension, control_dimension)

"Vectorize all players' states/controls for quadratic costs."
function all_xu(z₁, z₂, z₃)
	xs¹, us¹ = unpack(z₁)
	xs², us² = unpack(z₂)
	xs³, us³ = unpack(z₃)
	x = vcat(xs¹..., xs²..., xs³...)
	u = vcat(us¹..., us²..., us³...)
	return x, u
end

"Per-player discrete dynamics residuals stacked over t=1:T."
g(z) =
	mapreduce(vcat, 1:T) do t
		xs, us = unpack(z)
		xs[t+1] .- xs[t] .- Δt .* us[t]
	end

"Thin wrapper around PATH solve."
function solve_path(F, vars, θ; guess = nothing)
	z̲ = fill(-Inf, length(F))
	z̅ = fill(Inf, length(F))
	parametric = ParametricMCPs.ParametricMCP(F, vars, [θ], z̲, z̅; compute_sensitivities = false)
	z0 = isnothing(guess) ? zeros(length(vars)) : guess
	ParametricMCPs.solve(
		parametric, [1e-5];
		initial_guess = z0,
		verbose = false,
		cumulative_iteration_limit = 100_000,
		proximal_perturbation = 1e-2,
		use_basics = true,
		use_start = true,
	)
end

"Pretty print a player's (x,u) and objective."
function print_solution(label, z, J)
	xs, us = unpack(z)
	println("$label (x,u): ($xs, $us)")
	println("$label Objective: $(J)")
end

# ------------------------------------------------------------
# Symbolics & variables
# ------------------------------------------------------------
backend = SymbolicTracingUtils.SymbolicsBackend()
z₁ = SymbolicTracingUtils.make_variables(backend, Symbol("z̃1"), primal_dimension_per_player)
z₂ = SymbolicTracingUtils.make_variables(backend, Symbol("z̃2"), primal_dimension_per_player)
z₃ = SymbolicTracingUtils.make_variables(backend, Symbol("z̃3"), primal_dimension_per_player)
symbolic_type = eltype(z₁)
θ = only(SymbolicTracingUtils.make_variables(backend, :θ, 1))

# Initial conditions (as equality constraints)
xs¹, _ = unpack(z₁);
xs², _ = unpack(z₂);
xs³, _ = unpack(z₃)
ic₁ = xs¹[1] .- [0.0; 2.0] # (p_x, p_y) in meters INPUT from Tianyu
ic₂ = xs²[1] .- [2.0; 4.0]
ic₃ = xs³[1] .- [6.0; 8.0]

total_x_dim = N * state_dimension * (T + 1)
total_u_dim = N * control_dimension * (T + 1)

# # Random positive-definite style blocks (Q'Q, R'R are used)
# "Build one player’s quadratic cost from (Q,R)."
# function J_check(z₁, z₂, z₃, Q, R)
# 	x, u = all_xu(z₁, z₂, z₃)
# 	0.5 * x' * (Q' * Q) * x + 0.5 * u' * (R' * R) * u
# end
# Q¹, R¹ = rand(total_x_dim, total_x_dim), rand(total_u_dim, total_u_dim)
# Q², R² = rand(total_x_dim, total_x_dim), rand(total_u_dim, total_u_dim)
# Q³, R³ = rand(total_x_dim, total_x_dim), rand(total_u_dim, total_u_dim)
# J¹(z1,z2,z3) = J_check(z1, z2, z3, Q¹, R¹)
# J²(z1,z2,z3) = J_check(z1, z2, z3, Q², R²)
# J³(z1,z2,z3) = J_check(z1, z2, z3, Q³, R³)


#### player objectives ####
# player 3 (follower)'s objective function: P3 follows P2
function J₃(z₁, z₂, z₃)
	(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
	xs³, us³ = xs, us
	(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
	xs², us² = xs, us
	0.5*sum((xs³[end] .- xs²[end]) .^ 2) + 0.05*sum(sum(u³ .^ 2) for u³ in us³)
end

# player 2 (leader)'s objective function: P2 wants to get to the origin
function J₂(z₁, z₂, z₃)
	(; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
	xs³, us³ = xs, us
	(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
	xs², us² = xs, us
	(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
	xs¹, us¹ = xs, us
	# sum((0.5*(xs¹[end] .+ xs³[end])) .^ 2) + 0.05*sum(sum(u .^ 2) for u in us²)
	0.5*sum((xs²[end] .- xs¹[end]) .^ 2) + 0.05*sum(sum(u .^ 2) for u in us²)
end

# player 1 (top leader)'s objective function: P1 wants to get close to P2's final position
function J₁(z₁, z₂, z₃)
	(; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
	xs¹, us¹ = xs, us
	(; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
	xs², us² = xs, us
	# 0.5*sum((xs¹[end] .- xs²[end]) .^ 2) + 0.05*sum(sum(u .^ 2) for u in us¹)
	0.5*sum(xs¹[end] .^ 2) + 0.05*sum(sum(u .^ 2) for u in us¹)
end
# ------------------------------------------------------------
# Stackelberg (P1 leader; P2 middle-leader; P3 follower)
# ------------------------------------------------------------
# follower KKT (player 3)
λ₃ = SymbolicTracingUtils.make_variables(backend, Symbol("λ_3"), length(g(z₃)) + length(ic₃))
L₃ = J₃(z₁, z₂, z₃) - λ₃' * [g(z₃); ic₃]
π₃ = vcat(Symbolics.gradient(L₃, z₃), g(z₃), ic₃)

# middle-leader (player 2): include follower stationarity via implicit fn (ϕ³)
p₂_eq_dim   = length(g(z₂)) + length(ic₂)
p₂_stat_dim = length(z₃)
λ₂₁      = SymbolicTracingUtils.make_variables(backend, Symbol("λ_2_1"), p₂_eq_dim)
λ₂₂      = SymbolicTracingUtils.make_variables(backend, Symbol("λ_2_2"), p₂_stat_dim)
λ₂         = vcat(λ₂₁, λ₂₂)

M₂ = Symbolics.jacobian(π₃, [z₃; λ₃])
N₂ = Symbolics.jacobian(π₃, [z₁; z₂])

# Select the "z₃ rows" from the response [Δz₃; Δλ₃]
S₃ = hcat(I(length(z₃)), zeros(length(z₃), length(λ₃)))
ϕ³ = - S₃ * (M₂ \ N₂ * [z₁; z₂])

L² = J₂(z₁, z₂, z₃) - λ₂₁' * [g(z₂); ic₂] - λ₂₂' * (z₃ .- ϕ³)
π₂ = vcat(Symbolics.gradient(L², z₂),
	Symbolics.gradient(L², z₃),
	g(z₂), ic₂)

# leader (player 1): include P2/P3 stationarity via implicit fn (ϕ², ϕ³)
p₁_eq_dim  = length(g(z₁)) + length(ic₁)
p₁_stat₂ = length(z₂)
p₁_stat₃ = length(z₃)
λ₁₁     = SymbolicTracingUtils.make_variables(backend, Symbol("λ_1_1"), p₁_eq_dim)
λ₁₂     = SymbolicTracingUtils.make_variables(backend, Symbol("λ_1_2"), p₁_stat₂)
λ₁₃     = SymbolicTracingUtils.make_variables(backend, Symbol("λ_1_3"), p₁_stat₃)
λ₁        = vcat(λ₁₁, λ₁₂, λ₁₃)

M₁ = Symbolics.jacobian(π₂, [z₂; z₃; λ₂; λ₃])
N₁ = Symbolics.jacobian(π₂, z₁)

S₂ = hcat(I(length(z₂)), zeros(length(z₂), length(z₃) + length(λ₂) + length(λ₃)))
ϕ² = - S₂ * (M₁ \ N₁ * z₁)

L¹ = J₁(z₁, z₂, z₃) - λ₁₁' * [g(z₁); ic₁] - λ₁₂' * (z₂ .- ϕ²) - λ₁₃' * (z₃ .- ϕ³)
π₁ = vcat(Symbolics.gradient(L¹, z₁),
	Symbolics.gradient(L¹, z₂),
	Symbolics.gradient(L¹, z₃),
	g(z₁), ic₁)

# Final MCP (Stackelberg): leader KKT + middle-leader KKT + follower KKT
F_stack = Vector{symbolic_type}([π₁; π₂; π₃])
vars_stack = vcat(z₁, z₂, z₃, λ₁, λ₂, λ₃)

# Solve Stackelberg
z_all, status, info = solve_path(F_stack, vars_stack, θ)
@show status
z₁_sol, z₂_sol, z₃_sol = split3(z_all, length(z₁), length(z₂), length(z₃))

print_solution("P1", z₁_sol, J₁(z₁_sol, z₂_sol, z₃_sol))
print_solution("P2", z₂_sol, J₂(z₁_sol, z₂_sol, z₃_sol))
print_solution("P3", z₃_sol, J₃(z₁_sol, z₂_sol, z₃_sol))

# Helper: turn the vector-of-vectors `xs` into a 2×(T+1) matrix
state_matrix(xs_vec) = hcat(xs_vec...)  # each column is x at time t

# Reconstruct trajectories from solutions
xs1, us1 = unflatten_trajectory(z₁_sol, state_dimension, control_dimension)
xs2, us2 = unflatten_trajectory(z₂_sol, state_dimension, control_dimension)
xs3, us3 = unflatten_trajectory(z₃_sol, state_dimension, control_dimension)
# (v_x, v_y) in m/s OUTPUT to Tianyu

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


# ------------------------------------------------------------
# Nash comparison (all players’ KKT simultaneously)
# ------------------------------------------------------------
λₙ₁ = SymbolicTracingUtils.make_variables(backend, Symbol("λₙ_1"), length(g(z₁)) + length(ic₁))
λₙ₂ = SymbolicTracingUtils.make_variables(backend, Symbol("λₙ_2"), length(g(z₂)) + length(ic₂))
λₙ₃ = SymbolicTracingUtils.make_variables(backend, Symbol("λₙ_3"), length(g(z₃)) + length(ic₃))

Lₙ₁ = J₁(z₁, z₂, z₃) - λₙ₁' * [g(z₁); ic₁]
Lₙ₂ = J₂(z₁, z₂, z₃) - λₙ₂' * [g(z₂); ic₂]
Lₙ₃ = J₃(z₁, z₂, z₃) - λₙ₃' * [g(z₃); ic₃]

F_nash = Vector{symbolic_type}([
	Symbolics.gradient(Lₙ₁, z₁); g(z₁); ic₁;
	Symbolics.gradient(Lₙ₂, z₂); g(z₂); ic₂;
	Symbolics.gradient(Lₙ₃, z₃); g(z₃); ic₃;
])
vars_nash = vcat(z₁, z₂, z₃, λₙ₁, λₙ₂, λₙ₃)

z_all_n, status_n, info_n = solve_path(F_nash, vars_nash, θ)
@show status_n
zₙ₁_sol, zₙ₂_sol, zₙ₃_sol = split3(z_all_n, length(z₁), length(z₂), length(z₃))

print_solution("P1 (Nash)", zₙ₁_sol, J₁(zₙ₁_sol, zₙ₂_sol, zₙ₃_sol))
print_solution("P2 (Nash)", zₙ₂_sol, J₂(zₙ₁_sol, zₙ₂_sol, zₙ₃_sol))
print_solution("P3 (Nash)", zₙ₃_sol, J₃(zₙ₁_sol, zₙ₂_sol, zₙ₃_sol))

println("L1 difference: $(norm(zₙ₁_sol .- z₁_sol, 1)) + $(norm(zₙ₂_sol .- z₂_sol, 1)) + $(norm(zₙ₃_sol .- z₃_sol, 1))")