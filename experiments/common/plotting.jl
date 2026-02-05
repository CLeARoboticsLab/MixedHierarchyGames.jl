#=
    Plotting Utilities for Experiments

    Shared visualization functions for trajectory games.
    Requires Plots.jl to be loaded.
=#

using LinearAlgebra: norm

# ============================================================================
# Generic Trajectory Plotting
# ============================================================================

"""
    plot_trajectories_2d(trajectories; kwargs...)

Plot 2D trajectories for multiple players.

# Arguments
- `trajectories`: Vector of named tuples with `.xs` field (vector of state vectors)
- `labels`: Player labels (default: "P1", "P2", ...)
- `title`: Plot title
- `colors`: Vector of colors for each player
- `show_start_end`: Whether to mark start/end points
"""
function plot_trajectories_2d(
    trajectories;
    labels = ["P$i" for i in 1:length(trajectories)],
    title = "Player Trajectories",
    colors = [:red, :green, :blue, :orange, :purple, :cyan],
    markers = [:circle, :diamond, :utriangle, :star, :hexagon, :cross],
    show_start_end = true,
    aspect_ratio = :equal,
    kwargs...
)
    plt = plot(;
        xlabel = "x",
        ylabel = "y",
        title = title,
        legend = :outerright,
        aspect_ratio = aspect_ratio,
        grid = true,
        kwargs...
    )

    for (i, traj) in enumerate(trajectories)
        xs = traj.xs
        X = hcat(xs...)  # Convert to matrix [state_dim × T+1]

        c = colors[mod1(i, length(colors))]
        m = markers[mod1(i, length(markers))]

        # Plot trajectory
        plot!(plt, X[1, :], X[2, :];
            lw = 2, marker = m, ms = 4, color = c, label = labels[i])

        # Mark start and end
        if show_start_end
            scatter!(plt, [X[1, 1]], [X[2, 1]];
                markershape = :star5, ms = 8, color = c, label = "")
            scatter!(plt, [X[1, end]], [X[2, end]];
                markershape = :hexagon, ms = 8, color = c, label = "")
        end
    end

    return plt
end

"""
    plot_3player_trajectories(sols, state_dim, control_dim; kwargs...)

Plot trajectories for a 3-player game from solution vectors.
"""
function plot_3player_trajectories(
    sols::Vector{<:AbstractVector},
    state_dim::Int,
    control_dim::Int;
    T = nothing,
    Δt = nothing,
    unflatten_fn = nothing,
    kwargs...
)
    if unflatten_fn === nothing
        error("Must provide unflatten_fn (e.g., TrajectoryGamesBase.unflatten_trajectory)")
    end

    trajectories = [unflatten_fn(z, state_dim, control_dim) for z in sols]

    title = if T !== nothing && Δt !== nothing
        "Player Trajectories (T=$T, Δt=$Δt)"
    else
        "Player Trajectories"
    end

    plt = plot_trajectories_2d(trajectories; title = title, kwargs...)

    # Mark origin
    scatter!(plt, [0.0], [0.0];
        marker = :cross, ms = 8, color = :black, label = "Origin")

    return plt
end

# ============================================================================
# Lane Change Specific Plotting
# ============================================================================

"""
    plot_lane_change_trajectories(trajectories, R, T, Δt; kwargs...)

Plot trajectories for the lane change scenario with highway and on-ramp visualization.

# Arguments
- `trajectories`: Vector of trajectory named tuples with `.xs` field
- `R`: Turning radius / lane y-coordinate
- `T`: Time horizon
- `Δt`: Time step
"""
function plot_lane_change_trajectories(
    trajectories,
    R::Real,
    T::Integer,
    Δt::Real;
    labels = ["P1", "P2", "P3", "P4"],
    colors = [:red, :green, :blue, :orange],
    show = true,
    savepath = nothing,
    kwargs...
)
    plt = plot(;
        xlabel = "x",
        ylabel = "y",
        title = "Lane Change Trajectories (T=$T, Δt=$Δt)",
        legend = :outerright,
        aspect_ratio = :equal,
        grid = true,
        kwargs...
    )

    # Draw road infrastructure
    _draw_lane_change_road!(plt, R)

    # Plot each player's trajectory
    for (i, traj) in enumerate(trajectories)
        xs = traj.xs
        X = hcat(xs...)  # [state_dim × T+1]

        c = colors[mod1(i, length(colors))]

        # Trajectory line
        plot!(plt, X[1, :], X[2, :];
            lw = 2, marker = :circle, ms = 4, color = c, label = labels[i])

        # Start point
        scatter!(plt, [X[1, 1]], [X[2, 1]];
            markershape = :star5, ms = 8, color = c, label = "")

        # End point
        scatter!(plt, [X[1, end]], [X[2, end]];
            markershape = :hexagon, ms = 8, color = c, label = "")
    end

    if show
        display(plt)
    end

    if savepath !== nothing
        savefile = endswith(lowercase(String(savepath)), ".pdf") ? String(savepath) : String(savepath) * ".pdf"
        savefig(plt, savefile)
    end

    return plt
end

"""
Draw the lane change road infrastructure (highway + on-ramp).
"""
function _draw_lane_change_road!(plt, R)
    # Quarter circular on-ramp: from (-R, 0) to (0, R)
    θ = range(π, π/2; length = 300)

    r_main = R
    offsets = range(0.5, 1.0; length = length(θ))
    r_inner = R .- offsets
    r_outer = R .+ offsets

    x_main = r_main .* cos.(θ)
    y_main = r_main .* sin.(θ)
    x_inner = r_inner .* cos.(θ)
    y_inner = r_inner .* sin.(θ)
    x_outer = r_outer .* cos.(θ)
    y_outer = r_outer .* sin.(θ)

    # Center line (dotted)
    plot!(plt, x_main, y_main;
        lw = 1.0, linestyle = :dot, color = :black, label = "")

    # Lane boundaries
    plot!(plt, x_inner, y_inner;
        lw = 1.0, linestyle = :solid, color = :black, label = "")
    plot!(plt, x_outer, y_outer;
        lw = 1.0, linestyle = :solid, color = :black, label = "")

    # Horizontal highway
    hline_upper = R + 1.0
    hline_lower = R - 1.0
    plot!(plt, [-16, 4], [hline_upper, hline_upper];
        lw = 1.0, linestyle = :solid, color = :black, label = "")
    plot!(plt, [-16, 4], [hline_lower, hline_lower];
        lw = 1.0, linestyle = :solid, color = :black, label = "")

    return plt
end

# ============================================================================
# Distance Plotting
# ============================================================================

"""
    plot_pairwise_distances(trajectories, T, Δt; kwargs...)

Plot pairwise distances between players over time.
"""
function plot_pairwise_distances(
    trajectories,
    T::Integer,
    Δt::Real;
    labels = nothing,
    show = true,
    savepath = nothing,
    d_safe = nothing,
    kwargs...
)
    N = length(trajectories)
    time = collect(0:T) .* Δt

    # Extract position matrices (first 2 state components)
    Xs = [hcat(traj.xs...)[1:2, :] for traj in trajectories]

    plt = plot(;
        xlabel = "Time (s)",
        ylabel = "Distance",
        title = "Pairwise Player Distances",
        legend = :outerright,
        grid = true,
        kwargs...
    )

    # Plot all pairwise distances
    pair_idx = 1
    markers = [:circle, :diamond, :utriangle, :star, :hexagon, :cross, :square]

    for i in 1:N
        for j in (i+1):N
            dij = [norm(Xs[i][:, t] - Xs[j][:, t]) for t in 1:size(Xs[i], 2)]
            m = markers[mod1(pair_idx, length(markers))]
            plot!(plt, time, dij;
                lw = 2, marker = m, ms = 3, label = "d(P$i, P$j)")
            pair_idx += 1
        end
    end

    # Show safety distance threshold if provided
    if d_safe !== nothing
        hline!(plt, [d_safe]; lw = 2, linestyle = :dash, color = :red, label = "d_safe")
    end

    if show
        display(plt)
    end

    if savepath !== nothing
        savefile = endswith(lowercase(String(savepath)), ".pdf") ? String(savepath) : String(savepath) * ".pdf"
        savefig(plt, savefile)
    end

    return plt
end

# ============================================================================
# Pursuer-Protector-VIP Plotting
# ============================================================================

"""
    plot_pursuit_game(trajectories, x_goal; kwargs...)

Plot trajectories for the pursuer-protector-VIP game.
"""
function plot_pursuit_game(
    trajectories;
    x_goal = [0.0, 0.0],
    labels = ["Pursuer", "Protector", "VIP"],
    colors = [:red, :blue, :green],
    show = true,
    savepath = nothing,
    kwargs...
)
    plt = plot_trajectories_2d(
        trajectories;
        labels = labels,
        colors = colors,
        title = "Pursuer-Protector-VIP Game",
        kwargs...
    )

    # Mark goal
    scatter!(plt, [x_goal[1]], [x_goal[2]];
        marker = :star, ms = 12, color = :gold, label = "Goal")

    if show
        display(plt)
    end

    if savepath !== nothing
        savefile = endswith(lowercase(String(savepath)), ".pdf") ? String(savepath) : String(savepath) * ".pdf"
        savefig(plt, savefile)
    end

    return plt
end

# ============================================================================
# Solution Info Printing
# ============================================================================

"""
    print_solution_info(trajectories, Js, sols; verbose=true)

Print solution information for each player including trajectories and objective values.

# Arguments
- `trajectories`: Vector of trajectory named tuples with `.xs` and `.us` fields
- `Js`: Dict of objective functions (player index => function)
- `sols`: Vector of solution vectors for each player
- `verbose`: Whether to print detailed trajectory info
"""
function print_solution_info(trajectories, Js::Dict, sols::Vector; verbose=true)
    N = length(trajectories)

    println("\n" * "="^60)
    println("Solution Information")
    println("="^60)

    for i in 1:N
        traj = trajectories[i]
        println("\nPlayer $i:")
        if verbose
            println("  States (xs):")
            for (t, x) in enumerate(traj.xs)
                println("    t=$(t-1): $x")
            end
            println("  Controls (us):")
            for (t, u) in enumerate(traj.us)
                println("    t=$(t-1): $u")
            end
        end

        # Compute objective value
        if haskey(Js, i)
            obj_val = Js[i](sols..., nothing)
            println("  Objective value: $obj_val")
        end
    end
    println("="^60)
end

"""
    print_solution_info(sols, Js, state_dim, control_dim; unflatten_fn, verbose=true)

Print solution info from raw solution vectors.

# Arguments
- `sols`: Vector of solution vectors for each player
- `Js`: Dict of objective functions
- `state_dim`: State dimension
- `control_dim`: Control dimension
- `unflatten_fn`: Function to unflatten trajectory (e.g., TrajectoryGamesBase.unflatten_trajectory)
"""
function print_solution_info(
    sols::Vector{<:AbstractVector},
    Js::Dict,
    state_dim::Int,
    control_dim::Int;
    unflatten_fn,
    verbose=true
)
    trajectories = [unflatten_fn(z, state_dim, control_dim) for z in sols]
    print_solution_info(trajectories, Js, sols; verbose=verbose)
end

# ============================================================================
# Convergence Plotting
# ============================================================================

"""
    plot_convergence_summary(result; kwargs...)

Plot summary of convergence analysis results.
"""
function plot_convergence_summary(result; show = true, savepath = nothing)
    (; iterations, residuals, statuses, config) = result

    # Create subplot layout
    plt = plot(layout = (1, 2), size = (1000, 400))

    # Left: iterations histogram
    histogram!(plt[1], iterations;
        bins = 20,
        xlabel = "Iterations",
        ylabel = "Count",
        title = "Iteration Distribution",
        legend = false)

    # Right: residuals (log scale)
    scatter!(plt[2], 1:length(residuals), residuals;
        yscale = :log10,
        xlabel = "Run",
        ylabel = "Final Residual",
        title = "Final Residuals",
        marker = :circle,
        legend = false)

    if show
        display(plt)
    end

    if savepath !== nothing
        savefig(plt, savepath)
    end

    return plt
end
