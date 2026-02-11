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

Conditionally timed block. When `TIMING_ENABLED[]` is `true`, behaves like
`@timeit timer_output "label" expr`. When `false`, executes `expr` directly
with zero overhead (the timer output and label are not evaluated).

# Examples
```julia
to = TimerOutput()
enable_timing!()
@timeit_debug to "my section" begin
    # timed code
end
disable_timing!()
@timeit_debug to "my section" begin
    # NOT timed — zero overhead
end
```
"""
macro timeit_debug(timer_output, label, expr)
    quote
        if $TIMING_ENABLED[]
            @timeit $(esc(timer_output)) $(esc(label)) $(esc(expr))
        else
            $(esc(expr))
        end
    end
end
