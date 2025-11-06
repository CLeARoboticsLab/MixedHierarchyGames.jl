# include("examples/TestAutomaticSolver.jl") once before running this file

######### INPUT: Initial conditions ##########################
# parameters
R = 6.0  # turning radius
# x_goal = [1.5R; R; 0.0; 0.0]  # target position
# x0 = [
# 	[-2.0R; R; 1.0; 0.0], #[px, py, vx, vy]
# 	[-1.5R; R; 1.0; 0.0],
# 	[-R;  0.0; 0.0; 1.0],
# ]

# unicycle model initial conditions
x_goal = [1.5R; R; 0.0; 0.0]  # target position
x0 = [
	[-2.0R; R; 0.0; 2.0], #[px, py, ψ, v]
	[-1.5R; R; 0.0; 2.0],
	[-R; 0.0; pi/2; 1.523],
]

R = 6.0  # turning radius
T = 10 # 10
Δt = 0.5 # 0.5

function make_unicycle_traj(
    T::Integer,
    Δt::Real;
    R::Real = 6.0,
    split::Real = 0.5,
    x0::AbstractVector{<:Real} = [-R; 0.0; π/2; 1.523],  # default to your x0_3
)
    @assert Δt > 0 "Δt must be positive"
    @assert T ≥ 2 "Horizon T must be at least 2"
    @assert length(x0) == 4 "x0 must be [x, y, ψ, v]"

    # How many steps for the quarter circle vs straight segment
    T1 = Int(ceil(split * T))             # steps on the arc
    T1 = clamp(T1, 1, T-1)                # at least 1, leave ≥1 for straight
    T2 = T - T1

    # --- Build reference positions p_k = (x_k, y_k), k = 0..T ---
    # Quarter circle: θ from π down to π/2 in T1 equal steps (clockwise)
    Δθ = (π/2) / T1
    xs_pos = Vector{Float64}()
    ys_pos = Vector{Float64}()
    for k in 0:T1
        θ = π - k * Δθ
        push!(xs_pos, R * cos(θ))
        push!(ys_pos, R * sin(θ))
    end
    # Straight segment from (0, R) to (9, R) in T2 steps
    for s in 1:T2
        x = 9.0 * s / T2
        y = R
        push!(xs_pos, x)
        push!(ys_pos, y)
    end
    @assert length(xs_pos) == T + 1

    # --- Headings ψ_k along chord directions & speeds v_k matching step length ---
    ψ = Vector{Float64}(undef, T)   # heading used during step k: k = 0..T-1
    v = Vector{Float64}(undef, T)   # speed used during step k
    for k in 0:T-1
        dx = xs_pos[k+2] - xs_pos[k+1]
        dy = ys_pos[k+2] - ys_pos[k+1]
        ψ[k+1] = atan(dy, dx)
        v[k+1] = hypot(dx, dy) / Δt
    end

    # Extend terminal
    ψT = ψ[end]
    vT = v[end]

    # --- Controls to enforce discrete evolution of ψ and v exactly ---
    angle_diff = (a, b) -> atan(sin(a - b), cos(a - b))  # wrap(b→a)

    ω = Vector{Float64}(undef, T)
    a = Vector{Float64}(undef, T)
    for t in 1:T
        # for t == 1, use the GIVEN initial condition (x0) as "previous"
        ψ_prev = (t == 1) ? x0[3] : ψ[t-1]
        v_prev = (t == 1) ? x0[4] : v[t-1]

        ψ_curr = (t == T) ? ψT : ψ[t]
        v_curr = (t == T) ? vT : v[t]

        ω[t] = angle_diff(ψ_curr, ψ_prev) / Δt
        a[t] = (v_curr - v_prev) / Δt
    end

    # --- Assemble state and control arrays ---
    xs = Vector{Vector{Float64}}(undef, T + 1)
    us = Vector{Vector{Float64}}(undef, T)

    # 1) actual initial state = given x0
    xs[1] = collect(x0)

    # 2) intermediate states follow the reference positions but with our planned ψ,v
    for k in 2:T
        xs[k] = [xs_pos[k], ys_pos[k], ψ[k-1], v[k-1]]
    end

    # 3) terminal state
    xs[T+1] = [xs_pos[T+1], ys_pos[T+1], ψT, vT]

    # controls
    for t in 1:T
        us[t] = [a[t], ω[t]]
    end

    # Final padding for controls
    us = vcat(us, [[0.0, 0.0]])  #control at T+1

    return xs, us
end


x0_3 = [
	[-6.000000, 0.000000, 1.5707963268, 1.523600],  # t=0
	[-6.000000, 0.761800, 1.2566370614, 2.940800],  # t=1
	[-5.545600, 2.160300, 0.9424777961, 4.070200],  # t=2
	[-4.349400, 3.806800, 0.6283185307, 4.801200],  # t=3
	[-2.407200, 5.217800, 0.3141592653, 5.062200],  # t=4
	[0.000000, 6.000000, 0.0000000000, 3.600000],  # t=5  (end of arc)
	[1.800000, 6.000000, 0.0000000000, 3.600000],  # t=6
	[3.600000, 6.000000, 0.0000000000, 3.600000],  # t=7
	[5.400000, 6.000000, 0.0000000000, 3.600000],  # t=8
	[7.200000, 6.000000, 0.0000000000, 3.600000],  # t=9
	[9.000000, 6.000000, 0.0000000000, 3.600000],  # t=10 (final)
]

# Controls: u_t = [a, ω], t = 1..10 (length 10)
# First 5 steps: ω = -π/5, accelerations = 4× previous (to double speeds with Δt=0.5)
# Next 5 steps: ω = 0, a = 0 (constant v = 3.6)
u0_3 = [
	[+2.8344, -0.6283185307],
	[+2.2588, -0.6283185307],
	[+1.4620, -0.6283185307],
	[+0.5220, -0.6283185307],
	[-2.9244, -0.6283185307],
	[0.0000, 0.0000000000],
	[0.0000, 0.0000000000],
	[0.0000, 0.0000000000],
	[0.0000, 0.0000000000],
	[0.0000, 0.0000000000],
	[0.0000, 0.0000000000],
]

# Initial guess for all players
z0_guess_1_2 = zeros(6*(T+1) * 2) # for players 1 and 2
x0_3, u0_3 = make_unicycle_traj(T, Δt; R, split=0.5, x0 = x0[3])
z0_guess_3 = vcat([vcat(x0_3[t], u0_3[t]) for t in 1:T]...)
z0_guess = vcat(z0_guess_1_2, z0_guess_3)

Main.@infiltrate

###############################################################
# TestAutomaticSolver.nplayer_hierarchy_navigation(x0; verbose = false)
TestAutomaticSolver.nplayer_hierarchy_navigation_bicycle_dynamics(x0, x_goal, z0_guess, R, T, Δt; max_iters = 5000)