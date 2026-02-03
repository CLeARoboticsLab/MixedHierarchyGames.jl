#=
    Trajectory generation utilities for experiments

    Functions for generating initial guesses and reference trajectories.
=#

"""
    make_unicycle_traj(T, Δt; R=6.0, split=0.5, x0=[0.0, 0.0, π/2, 1.5])

Generate a dynamically feasible reference trajectory for unicycle model
that follows a quarter-circle arc then straight line.

# Arguments
- `T`: Time horizon (number of steps)
- `Δt`: Time step
- `R`: Turning radius
- `split`: Fraction of trajectory on circular arc (0 to 1)
- `x0`: Initial state [x, y, ψ, v]

# Returns
- `xs`: States trajectory (T+1 states)
- `us`: Controls trajectory (T+1 controls, last is zero)
"""
function make_unicycle_traj(
    T::Integer,
    Δt::Real;
    R::Real = 6.0,
    split::Real = 0.5,
    x0::AbstractVector{<:Real} = [-R, 0.0, π/2, 1.523],
)
    @assert Δt > 0 "Δt must be positive"
    @assert T >= 2 "Horizon T must be at least 2"
    @assert length(x0) == 4 "x0 must be [x, y, ψ, v]"

    # How many steps for the quarter circle vs straight segment
    T1 = Int(ceil(split * T))
    T1 = clamp(T1, 1, T-1)
    T2 = T - T1

    # Build reference positions
    Δθ = (π/2) / T1
    xs_pos = Float64[]
    ys_pos = Float64[]

    for k in 0:T1
        θ = π - k * Δθ
        push!(xs_pos, R * cos(θ))
        push!(ys_pos, R * sin(θ))
    end

    for s in 1:T2
        x = 9.0 * s / T2
        y = R
        push!(xs_pos, x)
        push!(ys_pos, y)
    end

    # Compute headings and speeds
    ψ = Vector{Float64}(undef, T)
    v = Vector{Float64}(undef, T)

    for k in 0:T-1
        dx = xs_pos[k+2] - xs_pos[k+1]
        dy = ys_pos[k+2] - ys_pos[k+1]
        ψ[k+1] = atan(dy, dx)
        v[k+1] = hypot(dx, dy) / Δt
    end

    ψT = ψ[end]
    vT = v[end]

    # Compute controls
    angle_diff = (a, b) -> atan(sin(a - b), cos(a - b))

    ω = Vector{Float64}(undef, T)
    a = Vector{Float64}(undef, T)

    for t in 1:T
        ψ_prev = (t == 1) ? x0[3] : ψ[t-1]
        v_prev = (t == 1) ? x0[4] : v[t-1]
        ψ_curr = (t == T) ? ψT : ψ[t]
        v_curr = (t == T) ? vT : v[t]

        ω[t] = angle_diff(ψ_curr, ψ_prev) / Δt
        a[t] = (v_curr - v_prev) / Δt
    end

    # Assemble trajectories
    xs = Vector{Vector{Float64}}(undef, T + 1)
    us = Vector{Vector{Float64}}(undef, T + 1)

    xs[1] = collect(Float64, x0)
    for k in 2:T
        xs[k] = [xs_pos[k], ys_pos[k], ψ[k-1], v[k-1]]
    end
    xs[T+1] = [xs_pos[T+1], ys_pos[T+1], ψT, vT]

    for t in 1:T
        us[t] = [a[t], ω[t]]
    end
    us[T+1] = [0.0, 0.0]

    return xs, us
end

"""
    make_straight_traj(T, Δt; x0=[0.0, 0.0, 0.0, 1.0])

Generate a dynamically feasible straight-line trajectory for unicycle model.

# Arguments
- `T`: Time horizon
- `Δt`: Time step
- `x0`: Initial state [x, y, ψ, v]

# Returns
- `xs`: States trajectory (T+1 states)
- `us`: Controls trajectory (T+1 controls)
"""
function make_straight_traj(
    T::Integer,
    Δt::Real;
    x0::AbstractVector{<:Real} = [0.0, 0.0, 0.0, 1.0],
)
    @assert Δt > 0 "Δt must be positive"
    @assert T >= 1 "Horizon T must be at least 1"
    @assert length(x0) == 4 "x0 must be [x, y, ψ, v]"

    ψ0 = Float64(x0[3])
    v0 = Float64(x0[4])

    xs = Vector{Vector{Float64}}(undef, T + 1)
    us = Vector{Vector{Float64}}(undef, T + 1)

    xs[1] = collect(Float64, x0)

    for k in 2:(T + 1)
        Δ = (k - 1) * Δt
        x = x0[1] + v0 * cos(ψ0) * Δ
        y = x0[2] + v0 * sin(ψ0) * Δ
        xs[k] = [x, y, ψ0, v0]
    end

    for t in 1:T
        us[t] = [0.0, 0.0]
    end
    us[T+1] = [0.0, 0.0]

    return xs, us
end

"""
    flatten_trajectory(xs, us)

Flatten state and control trajectories into single vector.

# Arguments
- `xs`: Vector of state vectors (length T+1)
- `us`: Vector of control vectors (length T or T+1)

# Returns
- `z`: Flattened vector [x_1; u_1; x_2; u_2; ...; x_{T+1}; u_{T+1}]
"""
function flatten_trajectory(xs, us)
    T = length(xs) - 1
    parts = Any[]
    for t in 1:T
        push!(parts, xs[t])
        push!(parts, us[t])
    end
    push!(parts, xs[end])
    if length(us) > T
        push!(parts, us[end])
    end
    return vcat(parts...)
end
