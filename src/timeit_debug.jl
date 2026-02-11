"""
    TIMING_ENABLED

Global flag controlling whether `@timeit_debug` records timing data.
When `false` (default), `@timeit_debug` expands to execute the body with zero overhead.
When `true`, `@timeit_debug` delegates to `@timeit` from TimerOutputs.jl.

Toggle with [`enable_timing!`](@ref) and [`disable_timing!`](@ref).
"""
const TIMING_ENABLED = Ref(false)

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
When disabled, `@timeit_debug` has zero overhead.
"""
function disable_timing!()
    TIMING_ENABLED[] = false
    return nothing
end

"""
    @timeit_debug timer_output "label" expr

Conditionally timed block. When `TIMING_ENABLED[]` is `true`, records
timing via `begin_timed_section!`/`end_timed_section!` from TimerOutputs.jl.
When `false`, executes `expr` directly with near-zero overhead (one branch check).

The body is never duplicated in the macro expansion, so function definitions
inside the body work correctly.

# Examples
```julia
to = TimerOutput()
enable_timing!()
@timeit_debug to "my section" begin
    # timed code
end
disable_timing!()
@timeit_debug to "my section" begin
    # NOT timed — near-zero overhead
end
```
"""
macro timeit_debug(timer_output, label, expr)
    _section = gensym("section")

    # Use Expr(:tryfinally, body, cleanup) to avoid creating a new scope
    # (same pattern as TimerOutputs.@timeit uses internally).
    setup = quote
        local $_section = if $TIMING_ENABLED[]
            TimerOutputs.begin_timed_section!($(esc(timer_output)), $(esc(label)))
        else
            nothing
        end
    end

    cleanup = quote
        if $_section !== nothing
            TimerOutputs.end_timed_section!($(esc(timer_output)), $_section)
        end
    end

    Expr(:block,
        setup,
        Expr(:tryfinally, esc(expr), cleanup))
end
