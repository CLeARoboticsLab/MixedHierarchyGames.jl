using Symbolics
using SymbolicTracingUtils
using LinearAlgebra: norm

"""
Evaluates the symbolic KKT conditions `πs` at the numerical solution `z_sol`.

This function substitutes the numerical values from `z_sol` and `parameter_value`
into the symbolic expressions for each player's KKT conditions and computes the
norm of the resulting residual vectors. These norms should be close to zero for a
valid solution.

# Arguments
- `πs::Dict{Int, Any}`: A dictionary mapping player index to its symbolic KKT conditions.
- `all_variables::Vector`: A vector of all symbolic decision variables.
- `z_sol::Vector`: The numerical solution vector corresponding to `all_variables`.
- `θs::Dict{Int, Any}`: A dictionary of symbolic parameter variables for each player.
- `parameter_values::Dict{Int, Float64}`: A dictionary of numerical values for the parameters `θs`.
- `tol::Float64`: Tolerance for checking if KKT conditions are satisfied.
- `verbose::Bool`: If true, prints the first few elements of each residual vector.
- `should_enforce::Bool`: If true, asserts that the KKT conditions are satisfied.

# Returns
- `Dict{Int, Float64}`: A dictionary mapping each player's index to the norm of their KKT residual.
"""
function evaluate_kkt_residuals(πs, all_variables, z_sol, θs, parameter_values; tol=1e-6, verbose = false, should_enforce = false)
	# Create list of symbolic output expressions.
	all_πs = Vector{Symbolics.Num}(vcat(collect(values(πs))...))

	all_vars_and_params = vcat(all_variables, vcat(collect(values(θs))...))

	# Build the in-place version of the function.
	π_fns! = SymbolicTracingUtils.build_function(all_πs, all_vars_and_params; in_place = true)

	# Allocate output storage (same size as all_πs).
	π_eval = similar(all_πs, Float64)

	# Evaluate in place
	π_fns!(π_eval, z_sol)

	if verbose
		println("\n" * "="^20 * " KKT Residuals " * "="^20)
		println("Are all KKT conditions satisfied? ", all(π_eval .< tol))
		if norm(π_eval) < tol
			println("KKT conditions are satisfied within tolerance! Norm: ", norm(π_eval))
		end
		println("="^55)
	end

	if should_enforce
		@assert norm(π_eval) < tol "KKT conditions not satisfied within tolerance. Norm: $(norm(π_eval)) > $tol"
	end

	return π_eval
end
