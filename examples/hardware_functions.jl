module HardwareFunctions

using Logging
using TimerOutputs
using TrajectoryGamesBase: unflatten_trajectory
using SymbolicTracingUtils

# Builds preoptimization info for the LQ example, with logging silenced by default.
function build_lq_preoptimization(T::Int=3, Δt::Float64=0.5; silence_logs::Bool=true,
                                  backend=SymbolicTracingUtils.SymbolicsBackend())
    # LQ example problem
    N, G, H, problem_dims, Js, gs = Main.get_three_player_openloop_lq_problem(T, Δt; verbose=false)

    # Silence all logs (including @info inside preoptimization)
    logger = silence_logs ? NullLogger() : current_logger()
        preopt = with_logger(logger) do
        Main.preoptimize_nonlq_solver(H, G, problem_dims.primal_dimension_per_player, Js, gs;
                                 backend, to=TimerOutput(), verbose=false)
    end

    # Return a wrapper with everything needed for RH solves
    return (; preopt, N, G, H, T, Δt, problem_dims, Js, gs)
end

# Receding-horizon navigation using precomputed preoptimization info.
# Returns the solution trajectory (states and controls) with all logs silenced by default.
function hardware_nplayer_hierarchy_navigation(pre, x0::Vector{<:AbstractVector}, z0_guess=nothing,
                                      tol::Float64=1e-6, max_iters::Int=30; silence_logs::Bool=true)
    logger = silence_logs ? NullLogger() : current_logger()
    return with_logger(logger) do
        N = pre.N
        Δt = pre.Δt
        state_dimension = pre.problem_dims.state_dimension
        control_dimension = pre.problem_dims.control_dimension
        ppp = pre.problem_dims.primal_dimension_per_player

        # Histories per player
        states_hist = [Vector{Vector{Float64}}() for _ in 1:N]
        controls_hist = [Vector{Vector{Float64}}() for _ in 1:N]
        for i in 1:N
            push!(states_hist[i], copy(x0[i]))
        end

        # Warm start
        z_guess = isnothing(z0_guess) ? zeros(length(pre.preopt.all_variables)) : z0_guess

        steps = hasproperty(pre, :T) ? pre.T : (hasproperty(pre, :H) && hasproperty(pre.H, :T) ? pre.H.T : 3)

        for k in 1:steps
            # One MPC solve using the precomputed preoptimization info
            z_sol, status, info, all_variables, vars, augmented =
                Main.run_nonlq_solver(pre.H, pre.G, ppp, pre.Js, pre.gs, z_guess;
                                 preoptimization_info=pre.preopt, parameter_value=1e-5,
                                 max_iters=max_iters, tol=tol, verbose=false, to=TimerOutput())

            # Split z_sol per player and extract current control and next state
            z_sols = Vector{Vector{Float64}}(undef, N)
            offs = 1
            for i in 1:N
                li = length(vars.zs[i])
                z_sols[i] = @view z_sol[offs:offs+li-1]
                offs += li
            end

            x_next = Vector{Vector{Float64}}(undef, N)
            u_curr = Vector{Vector{Float64}}(undef, N)
            for i in 1:N
                (; xs, us) = unflatten_trajectory(z_sols[i], state_dimension, control_dimension)
                u_curr[i] = us[1]
                x_next[i] = xs[2]
                push!(controls_hist[i], copy(u_curr[i]))
                push!(states_hist[i], copy(x_next[i]))
            end

            # Warm-start next iteration with the last solution
            z_guess = z_sol
        end

        return (; states=states_hist, controls=controls_hist)
    end


end

# Compatibility wrapper: PyCall sometimes converts a Python list-of-lists into a 2-D
# Julia Array (Matrix). Accept that and convert to the expected Vector{Vector} form.
function hardware_nplayer_hierarchy_navigation(pre, x0::AbstractMatrix{<:Real}, z0_guess=nothing,
                                              tol::Float64=1e-6, max_iters::Int=30; silence_logs::Bool=true)
    # Treat rows as player states: convert each row to a Vector{Float64}
    nrows = size(x0, 1)
    x0_vec = [Vector{Float64}(x0[i, :]) for i in 1:nrows]
    return hardware_nplayer_hierarchy_navigation(pre, x0_vec, z0_guess, tol, max_iters; silence_logs=silence_logs)
end

end # module