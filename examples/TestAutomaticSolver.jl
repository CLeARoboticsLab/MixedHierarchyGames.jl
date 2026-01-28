module TestAutomaticSolver

export approximate_solve_with_linsolve!, compute_K_evals, armijo_backtracking_linesearch,
       preoptimize_nonlq_solver, run_nonlq_solver, solve_nonlq_game_example,
       compare_lq_and_nonlq_solver, nplayer_hierarchy_navigation

include("test_automatic_solver.jl")

end
