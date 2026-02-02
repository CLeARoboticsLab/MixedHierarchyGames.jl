# This file confirms that the solutions found by Li 2024 et al.'s OLSE method is the same as the one for our method.
using LinearAlgebra
using BlockArrays
using Random

include("test_automatic_solver.jl")

function solve_siopt_stackelberg_with_solver(; T = 2, x0 = [1.0, 2.0, 2.0, 1.0], verbose = false)
	# Problem data (from OLSE_SIOPT_paper_example_2026.jl)
	nx = 4
	m = 2
	A = Matrix(1.0 * I(nx))
	B = Matrix(0.1 * I(nx))
	B1 = B[:, 1:2]
	B2 = B[:, 3:4]
	Q1 = 4.0 * [
		0 0 0 0;
		0 0 0 0;
		0 0 1 0;
		0 0 0 1;
	]
	Q2 = 4.0 * [
		1 0 -1 0;
		0 1 0 -1;
		-1 0 1 0;
		0 -1 0 1;
	]
	R1 = 2 * I(m)
	R2 = 2 * I(m)

	N = 2
	H = 1
	G = SimpleDiGraph(N)
	add_edge!(G, 1, 2)

	function unpack_u(z)
		us = Vector{Vector{eltype(z)}}(undef, T)
		for t in 1:T
			us[t] = z[(m * (t - 1) + 1):(m * t)]
		end
		return us
	end

	function rollout_x(u1, u2)
		xs = Vector{Vector{eltype(u1[1])}}(undef, T + 1)
		xs[1] = collect(x0)
		for t in 1:T
			xs[t + 1] = A * xs[t] + B1 * u1[t] + B2 * u2[t]
		end
		return xs
	end

	function J1(z1, z2, θ)
		u1 = unpack_u(z1)
		u2 = unpack_u(z2)
		xs = rollout_x(u1, u2)
		x_cost = sum(xs[t + 1]' * Q1 * xs[t + 1] for t in 1:T)
		u_cost = sum(u1[t]' * R1 * u1[t] for t in 1:T)
		return x_cost + u_cost
	end

	function J2(z1, z2, θ)
		u1 = unpack_u(z1)
		u2 = unpack_u(z2)
		xs = rollout_x(u1, u2)
		x_cost = sum(xs[t + 1]' * Q2 * xs[t + 1] for t in 1:T)
		u_cost = sum(u2[t]' * R2 * u2[t] for t in 1:T)
	return x_cost + u_cost
	end

	function compute_olse_solution()
		M2 = BlockArray(zeros((nx + m) * T + nx * T, (nx + m) * T + nx * T),
			vcat(m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T)),
			vcat(m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T)),
		)
		N2 = BlockArray(zeros((nx + m) * T + nx * T, nx + m * T),
			vcat(m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T)),
			vcat([nx], m * ones(Int, T)),
		)

		for t in 1:T
			M2[Block(t, t)] = R2
			M2[Block(t, t + T)] = -B2'
		end
		for t in 1:T
			M2[Block(t + T, t + 2 * T)] = Q2
			M2[Block(t + T, t + T)] = I(nx)
			if t > 1
				M2[Block(t + T - 1, t + T)] = -A'
			end
		end
		for t in 1:T
			M2[Block(t + 2 * T, t + 2 * T)] = I(nx)
			M2[Block(t + 2 * T, t)] = -B2
			if t > 1
				M2[Block(t + 2 * T, t + 2 * T - 1)] = -A
			end
			N2[Block(t + 2 * T, t + 1)] = -B1
		end
		N2[Block(2 * T + 1, 1)] = -A

		K2 = -inv(M2) * N2

		M = BlockArray(zeros(m * T + m * T + nx * T + nx * T + m * T,
			m * T + m * T + nx * T + nx * T + m * T),
			vcat(m * ones(Int, T), m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T), m * ones(Int, T)),
			vcat(m * ones(Int, T), m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T), m * ones(Int, T)),
		)
		N = BlockArray(zeros(m * T + m * T + nx * T + nx * T + m * T, nx),
			vcat(m * ones(Int, T), m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T), m * ones(Int, T)),
			[nx],
		)

		for t in 1:T
			M[Block(t, t)] = R1
			M[Block(t, t + 3 * T)] = -B1'
			M[Block(t, 1 + 4 * T)] = -K2[Block(t, 2)]'
			M[Block(t, 2 + 4 * T)] = -K2[Block(t, 3)]'
		end
		for t in 1:T
			M[Block(t + T, t + T)] = zeros(m, m)
			M[Block(t + T, t + 3 * T)] = -B2'
			M[Block(t + T, t + 4 * T)] = I(m)
		end
		for t in 1:T
			M[Block(t + 2 * T, t + 2 * T)] = Q1
			M[Block(t + 2 * T, t + 3 * T)] = I(nx)
			if t > 1
				M[Block(t + 2 * T - 1, t + 3 * T)] = -A'
			end
		end
		for t in 1:T
			M[Block(t + 3 * T, t + 2 * T)] = I(nx)
			M[Block(t + 3 * T, t)] = -B1
			M[Block(t + 3 * T, t + T)] = -B2
			if t > 1
				M[Block(t + 3 * T, t + 2 * T - 1)] = -A
			end
		end
		for t in 1:T
			M[Block(t + 4 * T, 1)] = -K2[Block(t, 2)]
			M[Block(t + 4 * T, 2)] = -K2[Block(t, 3)]
			M[Block(t + 4 * T, t + T)] = I(m)
			N[Block(t + 4 * T, 1)] = -K2[Block(t, 1)]
		end
		N[Block(3 * T + 1, 1)] = -A

		sol = -inv(M) * N * x0
		u1 = sol[1:(m * T)]
		u2 = K2[1:(m * T), 1:nx] * x0 + K2[1:(m * T), (nx + 1):(nx + m * T)] * u1

		u1_traj = unpack_u(u1)
		u2_traj = unpack_u(u2)
		xs = rollout_x(u1_traj, u2_traj)

		return (; u1_traj, u2_traj, xs)
	end

	Js = Dict{Int, Any}(1 => J1, 2 => J2)

	backend = SymbolicTracingUtils.SymbolicsBackend()
	θs = setup_problem_parameter_variables(backend, fill(length(x0), N); verbose = false)
	parameter_values = Dict(1 => x0, 2 => x0)

	gs = [z -> Symbolics.Num[] for _ in 1:N]
	primal_dimension_per_player = m * T

	z_sol, status, info, _, _, _ = run_nonlq_solver(
		H, G, primal_dimension_per_player, Js, gs, θs, parameter_values;
		max_iters = 50, tol = 1e-8, verbose = verbose,
	)

	u1_sol = z_sol[1:(m * T)]
	u2_sol = z_sol[(m * T + 1):(2 * m * T)]
	u1_traj = unpack_u(u1_sol)
	u2_traj = unpack_u(u2_sol)
	xs = rollout_x(u1_traj, u2_traj)
	olse = compute_olse_solution()

	u1_err = norm(vcat(u1_traj...) - vcat(olse.u1_traj...))
	u2_err = norm(vcat(u2_traj...) - vcat(olse.u2_traj...))
	x_err = norm(vcat(xs...) - vcat(olse.xs...))
	if verbose
		@info "Comparison vs OLSE" u1_err u2_err x_err
	end

	if verbose
		@info "status" status
		@info "info" info
	end

	return (; u1_traj, u2_traj, xs, status, info, olse, u1_err, u2_err, x_err)
end

if abspath(PROGRAM_FILE) == @__FILE__
	rng = MersenneTwister(0)
	for k in 1:10
		local x0 = randn(rng, 4)
		result = solve_siopt_stackelberg_with_solver(x0 = x0, verbose = false)
		@info "Run $(k)" x0 status = result.status u1_err = result.u1_err u2_err = result.u2_err x_err = result.x_err
	end
end
