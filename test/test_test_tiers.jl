@testset "Test Tier Configuration" begin
    # Test that the tier classification constants exist and are correct
    @testset "fast_test_files contains expected files" begin
        expected_fast = [
            "test_graph_utils.jl",
            "test_ordered_player_indices.jl",
            "test_symbolic_utils.jl",
            "test_problem_setup.jl",
            "test_qp_kkt.jl",
            "test_linesearch.jl",
            "test_input_validation.jl",
            "test_type_stability.jl",
            "test_dict_to_vector_storage.jl",
            "test_qp_solver.jl",
            "test_qp_failure_modes.jl",
            "test_interface.jl",
            "test_timer.jl",
            "test_timeit_debug.jl",
            "test_type_bounds.jl",
            "test_block_arrays.jl",
            "test_unified_interface.jl",
            "test_k_sym_dedup.jl",
            "olse/test_qp_solver.jl",
        ]
        @test Set(FAST_TEST_FILES) == Set(expected_fast)
    end

    @testset "slow_test_files contains expected files" begin
        expected_slow = [
            "test_nonlinear_solver_options.jl",
            "test_nonlinear_solver.jl",
            "test_kkt_verification.jl",
            "test_integration.jl",
            "test_flexible_callsite.jl",
            "olse/test_nonlinear_solver.jl",
            "test_sparse_solve.jl",
            "test_allocation_optimization.jl",
            "test_jacobian_buffer_safety.jl",
            "test_regularization.jl",
        ]
        @test Set(SLOW_TEST_FILES) == Set(expected_slow)
    end

    @testset "all test files are classified" begin
        all_classified = Set(vcat(FAST_TEST_FILES, SLOW_TEST_FILES))
        # Every test file in the original runtests should be classified
        expected_all = Set([
            "test_graph_utils.jl",
            "test_ordered_player_indices.jl",
            "test_symbolic_utils.jl",
            "test_problem_setup.jl",
            "test_qp_kkt.jl",
            "test_qp_solver.jl",
            "test_qp_failure_modes.jl",
            "test_linesearch.jl",
            "test_nonlinear_solver_options.jl",
            "test_nonlinear_solver.jl",
            "test_kkt_verification.jl",
            "test_interface.jl",
            "test_input_validation.jl",
            "test_integration.jl",
            "test_flexible_callsite.jl",
            "olse/test_qp_solver.jl",
            "olse/test_nonlinear_solver.jl",
            "test_type_stability.jl",
            "test_dict_to_vector_storage.jl",
            "test_type_bounds.jl",
            "test_block_arrays.jl",
            "test_timer.jl",
            "test_timeit_debug.jl",
            "test_sparse_solve.jl",
            "test_allocation_optimization.jl",
            "test_jacobian_buffer_safety.jl",
            "test_regularization.jl",
            "test_unified_interface.jl",
            "test_k_sym_dedup.jl",
        ])
        @test all_classified == expected_all
    end

    @testset "no overlap between fast and slow tiers" begin
        overlap = intersect(Set(FAST_TEST_FILES), Set(SLOW_TEST_FILES))
        @test isempty(overlap)
    end

    @testset "FAST_TESTS_ONLY controls which tests run" begin
        # When FAST_TESTS_ONLY is "true", only fast tests should be selected
        files_fast = get_test_files(true)
        @test Set(files_fast) == Set(FAST_TEST_FILES)

        # When FAST_TESTS_ONLY is not set, all tests should be selected
        files_all = get_test_files(false)
        @test Set(files_all) == Set(vcat(FAST_TEST_FILES, SLOW_TEST_FILES))
    end
end
