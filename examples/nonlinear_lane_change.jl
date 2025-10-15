# include("TestAutomaticSolver.jl")

######### INPUT: Initial conditions ##########################
x0 = [
	[0.0; 2.0], # [px, py]
	[2.0; 4.0],
	[6.0; 8.0],
]
###############################################################

# TODO: write nplayer_hierarchy_navigation_bicycle_dynamics(x0; verbose = false) inside module and call it here

TestAutomaticSolver.nplayer_hierarchy_navigation(x0; verbose = false)