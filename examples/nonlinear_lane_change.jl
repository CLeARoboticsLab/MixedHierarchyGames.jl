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

# Compute initial guess
T = 10
z0_guess_1_2 = zeros(66 * 2) # for players 1 and 2
# For player 3, create a circular arc trajectory as initial guess
# States: x_t = [x, y, ψ, v], t = 0..10 (length 11)
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
z0_guess_3 = vcat([vcat(x0_3[t], u0_3[t]) for t in 1:(T+1)]...)
z0_guess = vcat(z0_guess_1_2, z0_guess_3)

###############################################################


# TestAutomaticSolver.nplayer_hierarchy_navigation(x0; verbose = false)
TestAutomaticSolver.nplayer_hierarchy_navigation_bicycle_dynamics(x0, x_goal, z0_guess, R; max_iters = 350)