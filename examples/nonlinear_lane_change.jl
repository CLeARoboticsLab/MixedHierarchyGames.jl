# include("examples/TestAutomaticSolver.jl") once before running this file

######### INPUT: Initial conditions ##########################
x0 = [
	[0.0; 2.0; 1.0; 1.0], # [x, y, ψ, v]
	[2.0; 4.0; -1.0; 1.0],
	[6.0; 8.0; 0.0; 1.0],
]

###############################################################


# TestAutomaticSolver.nplayer_hierarchy_navigation(x0; verbose = false)
TestAutomaticSolver.nplayer_hierarchy_navigation_bicycle_dynamics(x0; verbose = false)