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
- `x_buffer::Union{Nothing,Vector}=nothing` - Pre-allocated buffer for trial points.
  When provided, avoids allocating `x + α*d` each iteration. Must have same length as `x`.

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
    x_buffer::Union{Nothing,Vector}=nothing,
)
    f_x = f(x)
    ϕ_0 = dot(f_x, f_x)

    x_new = something(x_buffer, similar(x))

    α = alpha_init
    for _ in 1:max_iters
        @. x_new = x + α * d
        f_new = f(x_new)
        ϕ_new = dot(f_new, f_new)

        # Sufficient decrease: ϕ(x + α*d) ≤ ϕ(x) + c1 * α * (-2 * ϕ(x))
        if ϕ_new <= ϕ_0 + c1 * α * (-2 * ϕ_0)
            return α
        end

        α *= rho
    end

    @warn lazy"Armijo line search failed to find sufficient decrease after $max_iters iterations"
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
- `x_buffer::Union{Nothing,Vector}=nothing` - Pre-allocated buffer for trial points.
  When provided, avoids allocating `x + α*d` each iteration. Must have same length as `x`.

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
    x_buffer::Union{Nothing,Vector}=nothing,
)
    f_x = f(x)
    ϕ_0 = dot(f_x, f_x)

    x_new = something(x_buffer, similar(x))

    α = alpha_init
    for _ in 1:max_iters
        @. x_new = x + α * d
        f_new = f(x_new)
        ϕ_new = dot(f_new, f_new)

        if ϕ_new < ϕ_0
            return α
        end

        α *= rho
    end

    @warn lazy"Geometric reduction line search failed to find decrease after $max_iters iterations"
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

"""
    armijo_quadratic_interp(f, x, d, alpha_init; c1=1e-4, rho=0.5, max_iters=20, x_buffer=nothing)

Armijo line search with quadratic interpolation (Nocedal & Wright §3.5).

Same interface as `armijo_backtracking`. On the first backtrack failure, fits a quadratic
through ϕ(0), ϕ'(0)≈-2ϕ(0), and ϕ(α₀) to predict the minimizer. Falls back to geometric
backtracking if the quadratic step also fails.

The quadratic minimizer is:
    α_quad = -ϕ'(0) * α² / (2 * [ϕ(α) - ϕ(0) - ϕ'(0)*α])

with safeguard: `clamp(α_quad, 0.1*α, 0.5*α)`.

# Arguments
- `f::Function` - Residual function evaluating at a point, returns a vector
- `x::Vector` - Current point
- `d::Vector` - Search direction (typically the Newton step)
- `alpha_init::Float64` - Initial step size

# Keyword Arguments
- `c1::Float64=1e-4` - Sufficient decrease parameter (Armijo constant)
- `rho::Float64=0.5` - Geometric fallback reduction factor
- `max_iters::Int=20` - Maximum number of backtracking iterations
- `x_buffer::Union{Nothing,Vector}=nothing` - Pre-allocated buffer for trial points

# Returns
- `α::Float64` - Selected step size, or `0.0` if no sufficient decrease found
"""
function armijo_quadratic_interp(
    f::Function,
    x::Vector,
    d::Vector,
    alpha_init::Float64;
    c1::Float64=1e-4,
    rho::Float64=0.5,
    max_iters::Int=20,
    x_buffer::Union{Nothing,Vector}=nothing,
)
    f_x = f(x)
    ϕ_0 = dot(f_x, f_x)
    # Directional derivative approximation for Newton-like methods
    dϕ_0 = -2 * ϕ_0

    x_new = something(x_buffer, similar(x))

    α = alpha_init

    # First trial
    @. x_new = x + α * d
    f_new = f(x_new)
    ϕ_α = dot(f_new, f_new)

    # Check Armijo condition
    if ϕ_α <= ϕ_0 + c1 * α * dϕ_0
        return α
    end

    # Quadratic interpolation: fit q(α) through ϕ(0), ϕ'(0), ϕ(α₀)
    # Minimizer: α_quad = -ϕ'(0) * α² / (2 * (ϕ(α) - ϕ(0) - ϕ'(0)*α))
    denom = 2 * (ϕ_α - ϕ_0 - dϕ_0 * α)
    if abs(denom) > 1e-30
        α_quad = -dϕ_0 * α^2 / denom
        # Safeguard: clamp to [0.1α, 0.5α]
        α = clamp(α_quad, 0.1 * α, 0.5 * α)
    else
        # Near-linear ϕ: fall back to geometric
        α *= rho
    end

    # Try quadratic step
    @. x_new = x + α * d
    f_new = f(x_new)
    ϕ_α = dot(f_new, f_new)

    if ϕ_α <= ϕ_0 + c1 * α * dϕ_0
        return α
    end

    # Geometric fallback for remaining iterations
    for _ in 3:max_iters
        α *= rho
        @. x_new = x + α * d
        f_new = f(x_new)
        ϕ_new = dot(f_new, f_new)

        if ϕ_new <= ϕ_0 + c1 * α * dϕ_0
            return α
        end
    end

    @warn lazy"Quadratic interpolation line search failed to find sufficient decrease after $max_iters iterations"
    return 0.0
end

"""
    armijo_interpolation(f, x, d, alpha_init; c1=1e-4, rho=0.5, max_iters=20, x_buffer=nothing)

Armijo line search with quadratic + cubic interpolation (Nocedal & Wright §3.5).

Same interface as `armijo_backtracking`. Uses quadratic interpolation after the first
backtrack failure, then cubic interpolation using the two most recent rejected points
for subsequent failures (eq 3.57-3.58).

# Arguments
- `f::Function` - Residual function evaluating at a point, returns a vector
- `x::Vector` - Current point
- `d::Vector` - Search direction (typically the Newton step)
- `alpha_init::Float64` - Initial step size

# Keyword Arguments
- `c1::Float64=1e-4` - Sufficient decrease parameter (Armijo constant)
- `rho::Float64=0.5` - Geometric fallback reduction factor
- `max_iters::Int=20` - Maximum number of backtracking iterations
- `x_buffer::Union{Nothing,Vector}=nothing` - Pre-allocated buffer for trial points

# Returns
- `α::Float64` - Selected step size, or `0.0` if no sufficient decrease found
"""
function armijo_interpolation(
    f::Function,
    x::Vector,
    d::Vector,
    alpha_init::Float64;
    c1::Float64=1e-4,
    rho::Float64=0.5,
    max_iters::Int=20,
    x_buffer::Union{Nothing,Vector}=nothing,
)
    f_x = f(x)
    ϕ_0 = dot(f_x, f_x)
    dϕ_0 = -2 * ϕ_0  # directional derivative approximation

    x_new = something(x_buffer, similar(x))

    # First trial
    α = alpha_init
    @. x_new = x + α * d
    f_new = f(x_new)
    ϕ_α = dot(f_new, f_new)

    if ϕ_α <= ϕ_0 + c1 * α * dϕ_0
        return α
    end

    # Track previous step for cubic interpolation
    α_prev = α
    ϕ_prev = ϕ_α

    # Second trial: quadratic interpolation
    denom = 2 * (ϕ_α - ϕ_0 - dϕ_0 * α)
    if abs(denom) > 1e-30
        α_quad = -dϕ_0 * α^2 / denom
        α = clamp(α_quad, 0.1 * α, 0.5 * α)
    else
        α *= rho
    end

    @. x_new = x + α * d
    f_new = f(x_new)
    ϕ_α = dot(f_new, f_new)

    if ϕ_α <= ϕ_0 + c1 * α * dϕ_0
        return α
    end

    # Subsequent trials: cubic interpolation using two most recent points
    for _ in 3:max_iters
        # Cubic interpolation (Nocedal & Wright eq 3.57-3.58)
        # Solve 2x2 system for coefficients a, b of cubic c(α) = a*α³ + b*α² + dϕ₀*α + ϕ₀
        d1 = ϕ_α - ϕ_0 - dϕ_0 * α
        d2 = ϕ_prev - ϕ_0 - dϕ_0 * α_prev
        denom_cubic = (α - α_prev) * α^2 * α_prev^2

        α_next = α * rho  # default fallback
        if abs(denom_cubic) > 1e-30
            a = (α_prev^2 * d1 - α^2 * d2) / denom_cubic
            b = (-α_prev^3 * d1 + α^3 * d2) / denom_cubic

            # Minimizer: α_min = (-b + √(b²-3aϕ'₀)) / (3a)
            disc = b^2 - 3 * a * dϕ_0
            if abs(a) > 1e-30 && disc >= 0.0
                α_cubic = (-b + sqrt(disc)) / (3 * a)
                α_next = clamp(α_cubic, 0.1 * α, 0.5 * α)
            else
                # Degenerate: a≈0 or negative discriminant → geometric fallback
                α_next = α * rho
            end
        end

        α_prev = α
        ϕ_prev = ϕ_α
        α = α_next

        @. x_new = x + α * d
        f_new = f(x_new)
        ϕ_α = dot(f_new, f_new)

        if ϕ_α <= ϕ_0 + c1 * α * dϕ_0
            return α
        end
    end

    @warn lazy"Interpolation line search failed to find sufficient decrease after $max_iters iterations"
    return 0.0
end
