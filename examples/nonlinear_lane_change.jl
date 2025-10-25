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

x_goal = [1.5R; R; 0.0; 0.0]  # target position
x0 = [
	[-2.0R; R; 0.0; 5.0], #[px, py, ψ, v]
	[-1.5R; R; 0.0; 5.0],
	[-R;  0.0; pi/2; 5.0],
]

###############################################################


# TestAutomaticSolver.nplayer_hierarchy_navigation(x0; verbose = false)
TestAutomaticSolver.nplayer_hierarchy_navigation_bicycle_dynamics(x0, x_goal, R; max_iters = 5000)