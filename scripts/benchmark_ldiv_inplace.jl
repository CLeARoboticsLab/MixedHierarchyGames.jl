#=
Benchmark: In-place ldiv! for K = M \ N allocation reduction

Measures allocation savings from using pre-allocated K_buffers with lu!/ldiv!
compared to the default M \ N path that allocates a new result each call.

Usage:
    julia --project=. scripts/benchmark_ldiv_inplace.jl
=#

using LinearAlgebra: I, norm, lu, lu!
using SparseArrays: sparse
using MixedHierarchyGames: _solve_K!

println("=" ^ 70)
println("Benchmark: In-place ldiv! for K = M \\ N")
println("=" ^ 70)

#==========================================================================
    Unit-level benchmark: _solve_K! allocation comparison
==========================================================================#

function benchmark_solve_K(n_rows, n_cols; n_iters=1000)
    M = randn(n_rows, n_rows)
    M = M' * M + 5I  # positive definite
    N = randn(n_rows, n_cols)
    K_buffer = Matrix{Float64}(undef, n_rows, n_cols)

    # Warmup (2 iterations each)
    for _ in 1:2
        _solve_K!(copy(M), copy(N), 1)
        _solve_K!(copy(M), copy(N), 1; K_buffer)
    end

    # --- Dense path ---
    alloc_without = 0
    t_without = @elapsed for _ in 1:n_iters
        alloc_without += @allocated _solve_K!(copy(M), copy(N), 1)
    end

    alloc_with = 0
    t_with = @elapsed for _ in 1:n_iters
        alloc_with += @allocated _solve_K!(copy(M), copy(N), 1; K_buffer)
    end

    avg_without = alloc_without / n_iters
    avg_with = alloc_with / n_iters
    savings = avg_without - avg_with
    pct = savings / avg_without * 100

    println("\n  Dense path ($(n_rows)×$(n_rows) M, $(n_rows)×$(n_cols) N):")
    println("    Without K_buffer: $(round(avg_without, digits=0)) bytes/call, $(round(t_without/n_iters*1e6, digits=1))μs/call")
    println("    With K_buffer:    $(round(avg_with, digits=0)) bytes/call, $(round(t_with/n_iters*1e6, digits=1))μs/call")
    println("    Savings:          $(round(savings, digits=0)) bytes/call ($(round(pct, digits=1))%)")
    println("    K matrix size:    $(n_rows * n_cols * 8) bytes")

    # --- Sparse path ---
    alloc_sp_without = 0
    t_sp_without = @elapsed for _ in 1:n_iters
        alloc_sp_without += @allocated _solve_K!(copy(M), copy(N), 1; use_sparse=true)
    end

    alloc_sp_with = 0
    t_sp_with = @elapsed for _ in 1:n_iters
        alloc_sp_with += @allocated _solve_K!(copy(M), copy(N), 1; K_buffer, use_sparse=true)
    end

    avg_sp_without = alloc_sp_without / n_iters
    avg_sp_with = alloc_sp_with / n_iters
    savings_sp = avg_sp_without - avg_sp_with
    pct_sp = savings_sp / avg_sp_without * 100

    println("\n  Sparse path ($(n_rows)×$(n_rows) M, $(n_rows)×$(n_cols) N):")
    println("    Without K_buffer: $(round(avg_sp_without, digits=0)) bytes/call, $(round(t_sp_without/n_iters*1e6, digits=1))μs/call")
    println("    With K_buffer:    $(round(avg_sp_with, digits=0)) bytes/call, $(round(t_sp_with/n_iters*1e6, digits=1))μs/call")
    println("    Savings:          $(round(savings_sp, digits=0)) bytes/call ($(round(pct_sp, digits=1))%)")

    # --- Dense path with regularization ---
    alloc_reg_without = 0
    t_reg_without = @elapsed for _ in 1:n_iters
        alloc_reg_without += @allocated _solve_K!(copy(M), copy(N), 1; regularization=1e-6)
    end

    alloc_reg_with = 0
    t_reg_with = @elapsed for _ in 1:n_iters
        alloc_reg_with += @allocated _solve_K!(copy(M), copy(N), 1; K_buffer, regularization=1e-6)
    end

    avg_reg_without = alloc_reg_without / n_iters
    avg_reg_with = alloc_reg_with / n_iters
    savings_reg = avg_reg_without - avg_reg_with
    pct_reg = savings_reg / avg_reg_without * 100

    println("\n  Dense + regularization path:")
    println("    Without K_buffer: $(round(avg_reg_without, digits=0)) bytes/call")
    println("    With K_buffer:    $(round(avg_reg_with, digits=0)) bytes/call")
    println("    Savings:          $(round(savings_reg, digits=0)) bytes/call ($(round(pct_reg, digits=1))%)")

    return (; avg_without, avg_with, savings, pct)
end

println("\n--- Small matrices (typical leaf player, ~10×10) ---")
benchmark_solve_K(10, 8)

println("\n--- Medium matrices (typical mid-chain player, ~30×30) ---")
benchmark_solve_K(30, 20)

println("\n--- Large matrices (lane-change scale, ~100×100) ---")
benchmark_solve_K(100, 60; n_iters=200)

println("\n" * "=" ^ 70)
println("Benchmark complete.")
