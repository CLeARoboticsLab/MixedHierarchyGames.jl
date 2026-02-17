#=
Benchmark script for Perf #7: Dict{Int, Any} type investigation in KKT construction.

Finding: All Dict{Int, Any} instances in qp_kkt.jl and nonlinear_kkt.jl are COLD PATH
(one-time symbolic construction). The hot-path containers were already typed:
- Vector{Union{Matrix{Float64}, Nothing}} for K/M/N evals
- Dict{Int, Matrix{Float64}} for M/N buffers
- Vector{MNFunctionWrapper} for compiled M/N functions
- Vector{Int} for π_sizes

This benchmark verifies that the hot-path dict access pattern (Dict{Int, Matrix{Float64}})
is not a bottleneck compared to the dominant cost of matrix operations.
=#

using MixedHierarchyGames
using Graphs: SimpleDiGraph, add_edge!
using BenchmarkTools

# Setup: 2-player Stackelberg hierarchy
G = SimpleDiGraph(2)
add_edge!(G, 1, 2)

state_dim = 4
control_dim = 2
T = 5
primal_dim = (state_dim + control_dim) * (T + 1)
primal_dims = fill(primal_dim, 2)

backend = default_backend()
θs = setup_problem_parameter_variables(fill(state_dim, 2); backend)
Δt = 0.1

Js = Dict(
    1 => (z1, z2; θ=nothing) -> begin
        (; xs, us) = MixedHierarchyGames.unflatten_trajectory(z1, state_dim, control_dim)
        sum(sum(x.^2) for x in xs) + 0.1 * sum(sum(u.^2) for u in us)
    end,
    2 => (z1, z2; θ=nothing) -> begin
        (; xs, us) = MixedHierarchyGames.unflatten_trajectory(z2, state_dim, control_dim)
        sum(sum(x.^2) for x in xs) + 0.1 * sum(sum(u.^2) for u in us)
    end
)

gs = [
    z -> begin
        (; xs, us) = MixedHierarchyGames.unflatten_trajectory(z, state_dim, control_dim)
        dyn = mapreduce(vcat, 1:T) do t; xs[t+1] - xs[t] - Δt * us[t] end
        ic = xs[1] - θs[1]
        vcat(dyn, ic)
    end,
    z -> begin
        (; xs, us) = MixedHierarchyGames.unflatten_trajectory(z, state_dim, control_dim)
        dyn = mapreduce(vcat, 1:T) do t; xs[t+1] - xs[t] - Δt * us[t] end
        ic = xs[1] - θs[2]
        vcat(dyn, ic)
    end
]

println("Building NonlinearSolver (one-time cost)...")
solver = NonlinearSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

initial_states = Dict(1 => randn(state_dim), 2 => randn(state_dim))

println("\n=== Benchmark: Full solve (includes hot-path typed containers) ===")
# Warmup
solve_raw(solver, initial_states; max_iters=5)

b = @benchmark solve_raw($solver, $initial_states; max_iters=20) samples=20
display(b)

println("\n=== Benchmark: compute_K_evals (hot-path dict access) ===")
problem_vars = solver.precomputed.problem_vars
setup_info = solver.precomputed.setup_info
z_current = zeros(length(solver.precomputed.all_variables))

# Pre-allocate buffers (as done in run_nonlinear_solver)
N_players = 2
M_buffers = Dict{Int, Matrix{Float64}}()
N_buffers = Dict{Int, Matrix{Float64}}()
k_eval_buffers = (;
    M_evals = Vector{Union{Matrix{Float64}, Nothing}}(nothing, N_players),
    N_evals = Vector{Union{Matrix{Float64}, Nothing}}(nothing, N_players),
    K_evals = Vector{Union{Matrix{Float64}, Nothing}}(nothing, N_players),
    follower_cache = Vector{Union{Vector{Int}, Nothing}}(nothing, N_players),
    buffer_cache = Vector{Union{Vector{Float64}, Nothing}}(nothing, N_players),
    all_K_vec = Vector{Float64}(undef, solver.precomputed.mcp_obj.parameter_dimension - sum(length(initial_states[k]) for k in keys(initial_states))),
)

# Warmup
compute_K_evals(z_current, problem_vars, setup_info; M_buffers, N_buffers, buffers=k_eval_buffers)

b2 = @benchmark compute_K_evals($z_current, $problem_vars, $setup_info;
    M_buffers=$M_buffers, N_buffers=$N_buffers, buffers=$k_eval_buffers) samples=100
display(b2)

println("\n=== Dict{Int, Matrix{Float64}} access micro-benchmark ===")
# Show that Dict access with concrete value types is fast
typed_dict = Dict{Int, Matrix{Float64}}(2 => randn(10, 10))
untyped_dict = Dict{Int, Any}(2 => randn(10, 10))

println("Typed Dict{Int, Matrix{Float64}} access:")
b3 = @benchmark $typed_dict[2]
display(b3)

println("\nUntyped Dict{Int, Any} access:")
b4 = @benchmark $untyped_dict[2]
display(b4)

println("\n=== Summary ===")
println("All Dict{Int, Any} in KKT construction are cold-path (one-time symbolic setup).")
println("Hot-path containers use concrete types: Vector{Union{Matrix{Float64}, Nothing}},")
println("Dict{Int, Matrix{Float64}}, Vector{MNFunctionWrapper}, Vector{Int}.")
println("No performance improvement from this PR — the hot path was already correctly typed.")
