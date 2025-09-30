using Symbolics
using SymbolicTracingUtils
using LinearAlgebra: norm

"""
	evaluate_kkt_residuals(πs, all_variables, z_sol, θ, parameter_value; verbose=false)

Evaluates the symbolic KKT conditions `πs` at the numerical solution `z_sol`.

This function substitutes the numerical values from `z_sol` and `parameter_value`
into the symbolic expressions for each player's KKT conditions and computes the
norm of the resulting residual vectors. These norms should be close to zero for a
valid solution.

# Arguments
- `πs::Dict{Int, Any}`: A dictionary mapping player index to its symbolic KKT conditions.
- `all_variables::Vector`: A vector of all symbolic decision variables.
- `z_sol::Vector`: The numerical solution vector corresponding to `all_variables`.
- `θ::SymbolicUtils.Symbolic`: The symbolic parameter.
- `parameter_value::Number`: The numerical value for the parameter `θ`.
- `verbose::Bool`: If true, prints the first few elements of each residual vector.

# Returns
- `Dict{Int, Float64}`: A dictionary mapping each player's index to the norm of their KKT residual.
"""
function evaluate_kkt_residuals(πs, all_variables, z_sol, θ, parameter_value; verbose = false)
	"""
	Evaluate the KKT conditions
	"""
	all_πs = Vector{Symbolics.Num}(vcat(collect(values(πs))...))
	# TODO: Run with inplace true to do the opposite of chun chun hee.
	π_fns = SymbolicTracingUtils.build_function(all_πs, all_variables; in_place = false)
	π_eval = π_fns(z_sol)

	println("\n" * "="^20 * " KKT Residuals " * "="^20)
	println("Are all KKT conditions satisfied? ", all(π_eval .< 1e-6))
	if norm(π_eval) < 1e-6
		println("KKT conditions are satisfied within tolerance! Norm: ", norm(π_eval))
	end
	println("="^55)

	return π_eval
end
