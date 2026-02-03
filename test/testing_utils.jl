#=
    Shared test utilities for MixedHierarchyGames test suite.
=#

using MixedHierarchyGames: make_symbolic_vector

"""
    make_θ(player::Int, dim::Int)

Convenience helper to create parameter vectors using SymbolicTracingUtils.
Equivalent to `make_symbolic_vector(:θ, player, dim)`.

Used in tests to create symbolic parameter variables without calling Symbolics directly.
"""
make_θ(player::Int, dim::Int) = make_symbolic_vector(:θ, player, dim)
