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
	# Order everything consistently by player index.
	order = sort(collect(keys(πs)))

	# Create list of symbolic output expressions.
	all_πs = Vector{Symbolics.Num}(vcat([πs[ii] for ii in order]...))

	θ_vec = vcat([θs[ii] for ii in order]...)
	all_vars_and_params = vcat(all_variables, θ_vec)

	if parameter_values isa AbstractVector
		if !isempty(parameter_values) && eltype(parameter_values) <: AbstractVector
			param_vals_vec = vcat(parameter_values...)
		else
			param_vals_vec = parameter_values
		end
	else
		per_player_param_vals = [parameter_values[ii] for ii in order]
		param_vals_vec = vcat(per_player_param_vals...)
	end
	@assert length(z_sol) == length(all_variables) "Expected z_sol length $(length(all_variables)); got $(length(z_sol))."
	@assert length(param_vals_vec) == length(θ_vec) "Expected θ length $(length(θ_vec)); got $(length(param_vals_vec))."

	# Substitute explicitly to avoid any argument-ordering mismatches.
	vals = Dict{Any, Any}()
	for (sym, val) in zip(all_variables, z_sol)
		vals[sym] = val
	end
	for (sym, val) in zip(θ_vec, param_vals_vec)
		vals[sym] = val
	end
	π_eval = Symbolics.substitute.(all_πs, Ref(vals))
	π_eval = Float64.(Symbolics.value.(π_eval))

	if verbose
		println("\n" * "="^20 * " KKT Residuals " * "="^20)
		for (idx, val) in enumerate(π_eval)
			if abs(val) >= tol
				println("  π[$idx] = $(val)    expr: $(all_πs[idx])")
			end
		end
		println("Are all KKT conditions satisfied? ", all(π_eval .< tol))
		println("‖π‖₂ = ", norm(π_eval))
		println("="^55)
	end

	if should_enforce
		@assert norm(π_eval) < tol "KKT conditions not satisfied within tolerance. Norm: $(norm(π_eval)) > $tol"
	end

	return π_eval
end
