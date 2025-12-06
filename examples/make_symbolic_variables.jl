"""
This file contains utility functions to define symbol strings in a consistent way for use in symbolic KKT solvers.
"""


# TODO: Turn this into an extension of the string type using my own type.
function make_symbolic_variable(args...)
    variable_name = args[1]
    time = string(last(args))

    time_str = "_" * time
    # Remove this line.
    time_str = ""

    num_items = length(args)

    @assert variable_name in [:x, :u, :λ, :ψ, :μ, :z, :M, :N, :Φ, :K, :θ]
    variable_name_str = string(variable_name)

    if variable_name in [:x, :θ] && num_items == 2 # Just :x
        return Symbol(variable_name_str * time_str)
    elseif variable_name in [:u, :λ, :z, :M, :N, :K] && num_items == 3
       return Symbol(variable_name_str * "^" * string(args[2]) * time_str)
    elseif variable_name in [:ψ, :μ] && num_items == 4
        return Symbol(variable_name_str * "^(" * string(args[2]) * "-" * string(args[3]) * ")" * time_str)
    elseif variable_name in [:z] && num_items > 3
        # For z variables, we assume the inputs are of the form (z, i, j, ..., t)
        indices = join(string.(args[2:num_items-1]), ",")
        return Symbol(variable_name_str * "^(" * indices * ")" *time_str)
    else
        error("Invalid format has number of args $(num_items) for $args.")
    end
end
