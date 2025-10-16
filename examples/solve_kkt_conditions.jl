"""
A file containing functions which, given a KKT system, solve for the primal and dual variables using various techniques.
"""

using ParametricMCPs: ParametricMCPs


"""Helper function to compute the step size `α` which solves:
				   α* = max(α ∈ [0, 1] : v + α δ ≥ (1 - τ) v).
"""
function fraction_to_the_boundary_linesearch(v, δ; τ = 0.995, decay = 0.5, tol = 1e-4)
	α = 1.0
	while any(@. v + α * δ < (1 - τ) * v)
		if α < tol
			return NaN
		end

		α *= decay
	end

	α
end



# Uses the PATH solver to solve general MCPs via ParametricMCPs.jl.

###### UTILS FOR PATH SOLVER  ######
# TODO: Fix to make it general based on whatever expressions are needed.
function solve_with_path(πs, variables, θ, parameter_value)
    symbolic_type = eltype(variables)
    # Final MCP vector: leader stationarity + leader constraints + follower KKT
    F = Vector{symbolic_type}([
        vcat(collect(values(πs))...)... # KKT conditions of all players
    ])

    z̲ = fill(-Inf, length(F));
    z̅ = fill(Inf, length(F))

    # Solve via PATH
    parametric_mcp = ParametricMCPs.ParametricMCP(F, variables, [θ], z̲, z̅; compute_sensitivities = false)
    z_sol, status, info = ParametricMCPs.solve(
        parametric_mcp,
        [parameter_value];
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
    return z_sol, status, info
end

function lq_game_linsolve(πs, variables, θ, parameter_value; to=TimerOutput(), linear_solve_algorithm = LinearSolve.UMFPACKFactorization(), verbose = false)
	"""
	Custom linear solver for LQ games using KKT conditions.
	Given KKT conditions π, the function solves for Newton step ∇F(z; ϵ) δz = -F(z; ϵ).

	Parameters
	----------
	πs: Dict{Int, Vector{Symbolics.Num}}
		Dictionary mapping player indices to their KKT conditions.
	variables: Vector{Symbolics.Num}
		Vector of all symbolic variables in the game.
	θ: Symbolics.Num
		Symbolic parameter for the game.
	parameter_value: Float64
		Numeric value to substitute for the symbolic parameter θ.
	to: TimerOutput
		TimerOutput object for profiling (default: new TimerOutput()).
	linear_solve_algorithm: LinearSolve algorithm (default: UMFPACKFactorization)
		Algorithm to use for the linear solve step (from LinearSolve.jl).
	verbose: Bool (default: false)
		Whether to print verbose output.

	Returns
	-------
	z: Vector{Float64}
		Solution vector for all variables.
	status: Symbol
		Status of the solver (:solved or :failed).
	"""

	symbolic_type = eltype(variables)
	# Final MCP vector: leader stationarity + leader constraints + follower KKT
	F = Vector{symbolic_type}([
		vcat(collect(values(πs))...)..., # KKT conditions of all players
	])

	z̲ = fill(-Inf, length(F));
	z̅ = fill(Inf, length(F))

	# Form mcp via PATH
	@timeit to "[LQ Solver][ParametricMCP Setup]" begin
		parametric_mcp = ParametricMCPs.ParametricMCP(F, variables, [θ], z̲, z̅; compute_sensitivities = false)
	end
	∇F = parametric_mcp.jacobian_z!.result_buffer
	F = zeros(length(F))
	δz = zeros(length(variables))

	# We use an arbitrary initial point for this LQ solver.
	arbitrary_init_pt = zeros(length(variables))

	@timeit to "[LQ Solver][LinearSolve.jl Setup]" begin
		linsolve = init(LinearProblem(∇F, δz), linear_solve_algorithm)
	end
	parametric_mcp.f!(F, arbitrary_init_pt, [parameter_value])
	parametric_mcp.jacobian_z!(∇F, arbitrary_init_pt, [parameter_value])
	linsolve.A = ∇F
	linsolve.b = -F
	solution = solve!(linsolve)

	linsolve_status = :solved
	if !SciMLBase.successful_retcode(solution) &&
		(solution.retcode !== SciMLBase.ReturnCode.Default)
		verbose &&
			@warn "Linear solve failed. Exiting prematurely. Return code: $(solution.retcode)"
		linsolve_status = :failed
	end

	show(to)

	z = solution.u

	return z, linsolve_status
end
