#=
    Common dynamics functions for experiments

    This file contains shared dynamics models used across experiments.
=#

using TrajectoryGamesBase: unflatten_trajectory

"""
    unicycle_dynamics(z, t; Δt, state_dim=4, control_dim=2)

Kinematic unicycle dynamics constraint (nonlinear).
State: x = [x, y, ψ, v] (position, heading, speed)
Control: u = [a, ω] (acceleration, yaw rate)

Returns the dynamics residual: x_{t+1} - f(x_t, u_t)
"""
function unicycle_dynamics(z, t; Δt, state_dim=4, control_dim=2)
    (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
    x_t = xs[t]
    u_t = us[t]
    x_tp1 = xs[t+1]

    x, y, ψ, v = x_t
    a, ω = u_t

    # Euler forward discretization
    xdot = v * cos(ψ)
    ydot = v * sin(ψ)
    psidot = ω
    vdot = a

    x_pred = x_t .+ Δt .* [xdot, ydot, psidot, vdot]

    return x_tp1 - x_pred
end

"""
    bicycle_dynamics(z, t; Δt, L=1.0, state_dim=4, control_dim=2)

Kinematic bicycle dynamics constraint (nonlinear).
State: x = [x, y, ψ, v] (position, heading, speed)
Control: u = [a, δ] (acceleration, steering angle)
L: wheelbase length

Returns the dynamics residual: x_{t+1} - f(x_t, u_t)
"""
function bicycle_dynamics(z, t; Δt, L=1.0, state_dim=4, control_dim=2)
    (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
    x_t = xs[t]
    u_t = us[t]
    x_tp1 = xs[t+1]

    x, y, ψ, v = x_t
    a, δ = u_t

    # Euler forward discretization
    xdot = v * cos(ψ)
    ydot = v * sin(ψ)
    psidot = (v / L) * tan(δ)
    vdot = a

    x_pred = x_t .+ Δt .* [xdot, ydot, psidot, vdot]

    return x_tp1 - x_pred
end

"""
    double_integrator_2d(z, t; Δt, state_dim=4, control_dim=2)

2D double integrator dynamics constraint (linear).
State: x = [x, y, vx, vy] (position, velocity)
Control: u = [ax, ay] (acceleration)

Returns the dynamics residual: x_{t+1} - f(x_t, u_t)
"""
function double_integrator_2d(z, t; Δt, state_dim=4, control_dim=2)
    (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)

    x_t = xs[t]
    u_t = us[t]
    x_tp1 = xs[t+1]

    x, y, vx, vy = x_t
    ax, ay = u_t

    x_next = x + Δt * vx + 0.5 * Δt^2 * ax
    y_next = y + Δt * vy + 0.5 * Δt^2 * ay
    vx_next = vx + Δt * ax
    vy_next = vy + Δt * ay

    x_pred = [x_next, y_next, vx_next, vy_next]
    return x_tp1 - x_pred
end

"""
    single_integrator_2d(z, t; Δt, state_dim=2, control_dim=2)

2D single integrator dynamics constraint (linear).
State: x = [x, y] (position)
Control: u = [vx, vy] (velocity)

Returns the dynamics residual: x_{t+1} - f(x_t, u_t)
"""
function single_integrator_2d(z, t; Δt, state_dim=2, control_dim=2)
    (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)

    x_t = xs[t]
    u_t = us[t]
    x_tp1 = xs[t+1]

    x_pred = x_t .+ Δt .* u_t
    return x_tp1 - x_pred
end
