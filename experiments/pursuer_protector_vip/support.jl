#=
    Pursuer-Protector-VIP - Support Functions

    Cost functions and experiment-specific helpers.
    Note: config.jl must be included before this file.
=#

using TrajectoryGamesBase: unflatten_trajectory

"""
    make_cost_functions(state_dim, control_dim, T, x_goal)

Create the cost functions for all players.

Returns a Dict mapping player index to cost function.
"""
function make_cost_functions(state_dim, control_dim, T, x_goal)
    # P1 (Pursuer): chase VIP, avoid protector, minimize control
    function J₁(z₁, z₂, z₃; θ=nothing)
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, us¹ = xs, us
        (; xs,) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs² = xs
        (; xs,) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³ = xs

        chase_vip = PURSUER_CHASE_WEIGHT * sum(sum((xs³[t] - xs¹[t]) .^ 2) for t in 1:T)
        avoid_protector = PURSUER_AVOID_WEIGHT * sum(sum((xs²[t] - xs¹[t]) .^ 2) for t in 1:T)
        control = PURSUER_CONTROL_WEIGHT * sum(sum(u .^ 2) for u in us¹)

        chase_vip + avoid_protector + control
    end

    # P2 (Protector/Leader): stay with VIP, keep VIP away from pursuer
    function J₂(z₁, z₂, z₃; θ=nothing)
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us
        (; xs,) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹ = xs
        (; xs,) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³ = xs

        stay_with_vip = PROTECTOR_STAY_WEIGHT * sum(sum((xs³[t] - xs²[t]) .^ 2) for t in 1:T)
        protect_vip = PROTECTOR_PROTECT_WEIGHT * sum(sum((xs³[t] - xs¹[t]) .^ 2) for t in 1:T)
        control = PROTECTOR_CONTROL_WEIGHT * sum(sum(u .^ 2) for u in us²)

        stay_with_vip + protect_vip + control
    end

    # P3 (VIP): reach goal, stay close to protector
    function J₃(z₁, z₂, z₃; θ=nothing)
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, us³ = xs, us
        (; xs,) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs² = xs

        reach_goal = VIP_GOAL_WEIGHT * sum((xs³[end] .- x_goal) .^ 2)
        stay_with_protector = VIP_STAY_WEIGHT * sum(sum((xs³[t] - xs²[t]) .^ 2) for t in 1:T)
        control = VIP_CONTROL_WEIGHT * sum(sum(u .^ 2) for u in us³)

        reach_goal + stay_with_protector + control
    end

    return Dict{Int,Any}(1 => J₁, 2 => J₂, 3 => J₃)
end
