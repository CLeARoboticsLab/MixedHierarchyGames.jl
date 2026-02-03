#=
    Collision avoidance cost functions for experiments

    Provides smooth approximations to collision avoidance constraints
    that work well with gradient-based solvers.
=#

"""
    smooth_collision(xsA, xsB; d_safe=2.0, α=20.0, w=1.0)

Smooth pairwise collision penalty between two trajectories.

Uses softplus approximation: cost = sum_t w * softplus(d_safe^2 - ||pA_t - pB_t||^2)^2

# Arguments
- `xsA`, `xsB`: Trajectories (vectors of state vectors)
- `d_safe`: Safe distance threshold
- `α`: Softplus sharpness (higher = sharper approximation)
- `w`: Cost weight

# Returns
- Total collision cost over trajectory
"""
function smooth_collision(xsA, xsB; d_safe=2.0, α=20.0, w=1.0)
    T = length(xsA)
    cost = zero(xsA[1][1])  # symbolic-friendly zero
    d_safe_sq = d_safe^2

    for k in 1:T
        # Use only position components (first 2 elements)
        Δp = xsA[k][1:2] .- xsB[k][1:2]
        d_sq = sum(Δp .^ 2)  # ||pA - pB||^2
        r = d_safe_sq - d_sq  # positive if too close
        h = (1/α) * log(1 + exp(α * r))  # softplus(r)
        cost += w * h^2
    end

    return cost
end

"""
    smooth_collision_all(xs_all...; d_safe=2.0, α=20.0, w=1.0)

Total pairwise collision cost over all player pairs.

# Arguments
- `xs_all...`: Trajectories for each player (variable number of arguments)
- `d_safe`: Safe distance threshold
- `α`: Softplus sharpness
- `w`: Cost weight per pair

# Returns
- Sum of collision costs over all unordered pairs (i, j) with i < j
"""
function smooth_collision_all(xs_all...; d_safe=2.0, α=20.0, w=1.0)
    N = length(xs_all)
    @assert N >= 2 "smooth_collision_all needs at least two players."

    total = 0.0

    # Sum over all unordered pairs (i < j)
    for i in 1:(N-1)
        for j in (i+1):N
            total += smooth_collision(xs_all[i], xs_all[j]; d_safe, α, w)
        end
    end

    return total
end

"""
    collision_constraint(xsA, xsB, t; d_safe=2.0)

Point-wise collision avoidance constraint at time t.

Returns ||pA_t - pB_t||^2 - d_safe^2 (positive when safe)
"""
function collision_constraint(xsA, xsB, t; d_safe=2.0)
    Δp = xsA[t][1:2] .- xsB[t][1:2]
    d_sq = sum(Δp .^ 2)
    return d_sq - d_safe^2
end
