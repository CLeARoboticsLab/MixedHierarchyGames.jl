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
    @show status
    return z_sol, status, info
end

# TODO: Add the LinSolve method here.