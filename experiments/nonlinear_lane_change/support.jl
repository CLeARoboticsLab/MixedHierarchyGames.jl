#=
    Nonlinear Lane Change - Support Functions

    Cost functions and experiment-specific helpers.
    Note: config.jl and common utilities must be included before this file.
=#

using TrajectoryGamesBase: unflatten_trajectory

"""
    make_cost_functions(state_dim, control_dim, T, R)

Create the cost functions for all players.

Returns a Dict mapping player index to cost function.
"""
function make_cost_functions(state_dim, control_dim, T, R)
    # P1 (Leader): stay in lane, avoid collisions, minimize control
    function J₁(z₁, z₂, z₃, z₄; θ=nothing)
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, us¹ = xs, us
        (; xs,) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs² = xs
        (; xs,) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³ = xs
        (; xs,) = unflatten_trajectory(z₄, state_dim, control_dim)
        xs⁴ = xs

        control = CONTROL_WEIGHT_P1 * sum(sum(u .^ 2) for u in us¹)
        collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
        velocity = VELOCITY_WEIGHT * sum((x¹[4] - TARGET_VELOCITY)^2 for x¹ in xs¹)
        y_deviation = Y_DEVIATION_WEIGHT * sum((x¹[2] - R)^2 for x¹ in xs¹)
        zero_heading = HEADING_WEIGHT * sum((x¹[3])^2 for x¹ in xs¹)

        control + collision + y_deviation + zero_heading + velocity
    end

    # P2 (Follower of P1): stay in lane, avoid collisions
    function J₂(z₁, z₂, z₃, z₄; θ=nothing)
        (; xs,) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹ = xs
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us
        (; xs,) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³ = xs
        (; xs,) = unflatten_trajectory(z₄, state_dim, control_dim)
        xs⁴ = xs

        control = CONTROL_WEIGHT_P2 * sum(sum(u .^ 2) for u in us²)
        collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
        velocity = VELOCITY_WEIGHT * sum((x²[4] - TARGET_VELOCITY)^2 for x² in xs²)
        y_deviation = Y_DEVIATION_WEIGHT * sum((x²[2] - R)^2 for x² in xs²)
        zero_heading = HEADING_WEIGHT * sum((x²[3])^2 for x² in xs²)

        control + collision + y_deviation + zero_heading + velocity
    end

    # P3 (Lane Merger - Nash): follow quarter circle then merge
    function J₃(z₁, z₂, z₃, z₄; θ=nothing)
        (; xs,) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹ = xs
        (; xs,) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs² = xs
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, us³ = xs, us
        (; xs,) = unflatten_trajectory(z₄, state_dim, control_dim)
        xs⁴ = xs

        # Track quarter circle for first half (stay on circular path of radius R)
        tracking = TRACKING_WEIGHT_P3 * sum((sum(x³[1:2] .^ 2) - R^2)^2 for x³ in xs³[2:div(T, 2)])
        control = CONTROL_WEIGHT_P3 * sum(sum(u³ .^ 2) for u³ in us³)
        collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
        velocity = VELOCITY_WEIGHT * sum((x³[4] - TARGET_VELOCITY)^2 for x³ in xs³)
        # Second half: merge into lane
        y_deviation = Y_DEVIATION_WEIGHT_P3 * sum((x³[2] - R)^2 for x³ in xs³[div(T, 2):T])
        zero_heading = HEADING_WEIGHT * sum((x³[3])^2 for x³ in xs³[div(T, 2):T])

        tracking + control + collision + y_deviation + zero_heading + velocity
    end

    # P4 (Follower of P2): stay in lane, avoid collisions
    function J₄(z₁, z₂, z₃, z₄; θ=nothing)
        (; xs,) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹ = xs
        (; xs,) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs² = xs
        (; xs,) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³ = xs
        (; xs, us) = unflatten_trajectory(z₄, state_dim, control_dim)
        xs⁴, us⁴ = xs, us

        control = CONTROL_WEIGHT_P4 * sum(sum(u .^ 2) for u in us⁴)
        collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
        velocity = VELOCITY_WEIGHT * sum((x⁴[4] - TARGET_VELOCITY)^2 for x⁴ in xs⁴)
        y_deviation = sum((x⁴[2] - R)^2 for x⁴ in xs⁴)  # Lower weight for P4
        zero_heading = HEADING_WEIGHT * sum((x⁴[3])^2 for x⁴ in xs⁴)

        control + collision + y_deviation + zero_heading + velocity
    end

    return Dict{Int,Any}(1 => J₁, 2 => J₂, 3 => J₃, 4 => J₄)
end

"""
    build_initial_guess(x0, R, T, Δt)

Build initial trajectory guess for all players.
"""
function build_initial_guess(x0, R, T, Δt)
    # P1, P2, P4: straight trajectories (already in lane)
    x0_1, u0_1 = make_straight_traj(T, Δt; x0=x0[1])
    x0_2, u0_2 = make_straight_traj(T, Δt; x0=x0[2])
    x0_4, u0_4 = make_straight_traj(T, Δt; x0=x0[4])

    # P3: unicycle trajectory (merging from ramp)
    x0_3, u0_3 = make_unicycle_traj(T, Δt; R, split=0.5, x0=x0[3])

    z0_guess_1 = flatten_trajectory(x0_1, u0_1)
    z0_guess_2 = flatten_trajectory(x0_2, u0_2)
    z0_guess_3 = flatten_trajectory(x0_3, u0_3)
    z0_guess_4 = flatten_trajectory(x0_4, u0_4)

    return vcat(z0_guess_1, z0_guess_2, z0_guess_3, z0_guess_4)
end
