#=
    LQ Three Player Chain - Support Functions

    Cost functions and experiment-specific helpers.
    Note: config.jl must be included before this file.
=#

using TrajectoryGamesBase: unflatten_trajectory

"""
    make_cost_functions(state_dim, control_dim)

Create the cost functions for all players.

Returns a Dict mapping player index to cost function.
"""
function make_cost_functions(state_dim, control_dim)
    # P1: get close to P2's final position, minimize control
    function J₁(z₁, z₂, z₃; θ=nothing)
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, us¹ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², _ = xs, us

        terminal = TERMINAL_WEIGHT * sum((xs¹[end] .- xs²[end]) .^ 2)
        control = CONTROL_WEIGHT * sum(sum(u .^ 2) for u in us¹)
        terminal + control
    end

    # P2 (leader): wants P1 and P3 to reach origin, minimize control
    function J₂(z₁, z₂, z₃; θ=nothing)
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, _ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, _ = xs, us

        # P2 wants average of P1 and P3 at origin
        terminal = sum((0.5 * (xs¹[end] .+ xs³[end])) .^ 2)
        control = CONTROL_WEIGHT * sum(sum(u .^ 2) for u in us²)
        terminal + control
    end

    # P3: get close to P2's final position, minimize own + P2's control
    function J₃(z₁, z₂, z₃; θ=nothing)
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, us³ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us

        terminal = TERMINAL_WEIGHT * sum((xs³[end] .- xs²[end]) .^ 2)
        control_self = CONTROL_WEIGHT * sum(sum(u .^ 2) for u in us³)
        control_p2 = CONTROL_WEIGHT * sum(sum(u .^ 2) for u in us²)
        terminal + control_self + control_p2
    end

    return Dict{Int,Any}(1 => J₁, 2 => J₂, 3 => J₃)
end
