"""
    armijo_backtracking(f, x, d, alpha_init; c1=1e-4, rho=0.5, max_iters=20)

Armijo backtracking line search for step size selection.

Uses the merit function ϕ(x) = ||f(x)||² and checks the sufficient decrease condition:

    ϕ(x + α*d) ≤ ϕ(x) + c1 * α * ∇ϕ'*d

where for Newton-like methods ∇ϕ'*d ≈ -2*||f(x)||².

# Arguments
- `f::Function` - Residual function evaluating at a point, returns a vector
- `x::Vector` - Current point
- `d::Vector` - Search direction (typically the Newton step)
- `alpha_init::Float64` - Initial step size

# Keyword Arguments
- `c1::Float64=1e-4` - Sufficient decrease parameter (Armijo constant)
- `rho::Float64=0.5` - Step size reduction factor per backtracking iteration
- `max_iters::Int=20` - Maximum number of backtracking iterations

# Returns
- `α::Float64` - Selected step size, or `0.0` if no sufficient decrease found
"""
function armijo_backtracking(
    f::Function,
    x::Vector,
    d::Vector,
    alpha_init::Float64;
    c1::Float64=1e-4,
    rho::Float64=0.5,
    max_iters::Int=20,
)
    f_x = f(x)
    ϕ_0 = norm(f_x)^2

    α = alpha_init
    for _ in 1:max_iters
        x_new = x .+ α .* d
        ϕ_new = norm(f(x_new))^2

        # Sufficient decrease: ϕ(x + α*d) ≤ ϕ(x) + c1 * α * (-2 * ϕ(x))
        if ϕ_new <= ϕ_0 + c1 * α * (-2 * ϕ_0)
            return α
        end

        α *= rho
    end

    @warn "Armijo line search failed to find sufficient decrease after $max_iters iterations"
    return 0.0
end

"""
    geometric_reduction(f, x, d, alpha_init; rho=0.5, max_iters=20)

Geometric step-size reduction line search.

Reduces the step size by a fixed factor `rho` each iteration until the merit function
ϕ(x) = ||f(x)||² strictly decreases:

    ϕ(x + α*d) < ϕ(x)

This is a simpler alternative to `armijo_backtracking` — it requires only strict decrease
rather than sufficient decrease, and has no Armijo constant `c1`.

# Arguments
- `f::Function` - Residual function evaluating at a point, returns a vector
- `x::Vector` - Current point
- `d::Vector` - Search direction (typically the Newton step)
- `alpha_init::Float64` - Initial step size

# Keyword Arguments
- `rho::Float64=0.5` - Step size reduction factor per iteration
- `max_iters::Int=20` - Maximum number of reduction iterations

# Returns
- `α::Float64` - Selected step size, or `0.0` if no decrease found
"""
function geometric_reduction(
    f::Function,
    x::Vector,
    d::Vector,
    alpha_init::Float64;
    rho::Float64=0.5,
    max_iters::Int=20,
)
    ϕ_0 = norm(f(x))^2

    α = alpha_init
    for _ in 1:max_iters
        x_new = x .+ α .* d
        ϕ_new = norm(f(x_new))^2

        if ϕ_new < ϕ_0
            return α
        end

        α *= rho
    end

    @warn "Geometric reduction line search failed to find decrease after $max_iters iterations"
    return 0.0
end

"""
    constant_step(alpha)

Create a constant step-size line search that always returns `alpha`.

Returns a closure with the same interface as other line search methods
`(f, x, d, alpha_init) -> α`, but ignores all arguments and returns the fixed step size.

Useful as a baseline or when the appropriate step size is known a priori.

# Arguments
- `alpha::Float64` - The fixed step size to return

# Returns
- A function `(f, x, d, alpha_init) -> alpha` that always returns the fixed step size
"""
function constant_step(alpha::Float64)
    return (f, x, d, alpha_init) -> alpha
end
