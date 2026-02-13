"""
    TIMING_ENABLED

Global flag controlling whether `@timeit_debug` records timing data.
When `false` (default), `@timeit_debug` executes the body with near-zero overhead
(one atomic boolean check + try/finally frame).
When `true`, `@timeit_debug` delegates to TimerOutputs.jl for full instrumentation.

Toggle with [`enable_timing!`](@ref) and [`disable_timing!`](@ref).

!!! note "Thread safety"
    Uses `Threads.Atomic{Bool}` for safe concurrent access. However, there is an
    inherent TOCTOU gap: timing state may change between the check and the execution
    of the body. This is acceptable for a profiling tool.
"""
const TIMING_ENABLED = Threads.Atomic{Bool}(false)

"""
    enable_timing!()

Enable timing instrumentation for all `@timeit_debug` calls.
"""
function enable_timing!()
    TIMING_ENABLED[] = true
    return nothing
end

"""
    disable_timing!()

Disable timing instrumentation for all `@timeit_debug` calls (default).
When disabled, `@timeit_debug` has near-zero overhead (one atomic boolean check).
"""
function disable_timing!()
    TIMING_ENABLED[] = false
    return nothing
end

"""
    is_timing_enabled() -> Bool

Check whether timing instrumentation is currently enabled.
"""
is_timing_enabled() = TIMING_ENABLED[]

"""
    with_timing(f)

Execute `f()` with timing enabled, restoring the previous state afterward.

!!! note "Thread safety"
    The read-then-write of `TIMING_ENABLED` is not atomic. Concurrent calls to
    `with_timing` from multiple threads may incorrectly restore the flag. This is
    acceptable for a profiling tool — avoid enabling/disabling timing from multiple
    threads simultaneously.

# Example
```julia
using TimerOutputs  # Required: MixedHierarchyGames does not re-export TimerOutput
to = TimerOutput()
with_timing() do
    solve(solver, parameter_values; to)
end
show(to)
```
"""
function with_timing(f::Function)
    was_enabled = TIMING_ENABLED[]
    enable_timing!()
    try
        f()
    finally
        TIMING_ENABLED[] = was_enabled
    end
end

"""
    @timeit_debug timer_output "label" expr

Conditionally timed block. When `TIMING_ENABLED[]` is `true`, records timing via
TimerOutputs.jl's `begin_timed_section!`/`end_timed_section!` API with proper
exception handling. When `false`, executes `expr` directly with near-zero overhead.

Returns the value of `expr` in both modes.

!!! note "Implementation"
    Uses `Expr(:tryfinally)` to keep the body in a single location in the AST.
    This avoids method-redefinition errors when `expr` contains function definitions
    (e.g., closures inside `@timeit_debug` blocks). The try/finally overhead when
    timing is disabled is negligible for the solver blocks this wraps (millisecond-scale
    operations, not tight inner loops).

# Examples
```julia
to = TimerOutput()
enable_timing!()
result = @timeit_debug to "my section" begin
    expensive_computation()  # timed
end
disable_timing!()
result = @timeit_debug to "my section" begin
    expensive_computation()  # NOT timed — near-zero overhead
end
```
"""
macro timeit_debug(timer_output, label, expr)
    _section = gensym("section")

    # Check the atomic flag and optionally begin a timed section.
    # When timing is disabled, _section is `nothing` and the try/finally
    # cleanup is a single `nothing !== nothing` branch (~1ns).
    setup = quote
        local $_section = if $(esc(TIMING_ENABLED))[]
            TimerOutputs.begin_timed_section!($(esc(timer_output)), $(esc(label)))
        else
            nothing
        end
    end

    # Properly close the timed section even if expr throws.
    cleanup = quote
        if $_section !== nothing
            TimerOutputs.end_timed_section!($(esc(timer_output)), $_section)
        end
    end

    # Use Expr(:tryfinally) instead of quote try...finally...end to avoid
    # double-escaping issues with nested quoting. The body appears exactly once
    # in the expansion, preventing method-redefinition errors for function
    # definitions inside the block.
    Expr(:block,
        setup,
        Expr(:tryfinally, esc(expr), cleanup))
end
