using Test

include("../examples/test_automatic_solver.jl")

@testset "Minimal Solver Test" begin
    # Test with simple initial conditions
    x0_test = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0]
    ]
    
    println("Running solver with initial conditions: ", x0_test)
    
    # Run the solver
    next_state, curr_control = nplayer_hierarchy_navigation(x0_test; verbose=true)
    
    # Basic checks
    @test length(next_state) == 3
    @test length(curr_control) == 3
    @test all(length(s) == 2 for s in next_state)
    @test all(length(u) == 2 for u in curr_control)
    
    # Check that values are finite
    @test all(isfinite.(vcat(next_state...)))
    @test all(isfinite.(vcat(curr_control...)))
    
    println("✓ Solver test passed!")
    println("Next states: ", next_state)
    println("Current controls: ", curr_control)
end

@testset "Multiple Initial Conditions" begin
    # Test with different initial conditions
    test_cases = [
        [[0.0, 2.0], [2.0, 4.0], [6.0, 8.0]],
        [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]],
        [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]
    ]
    
    for (i, x0) in enumerate(test_cases)
        println("\nTest case $i with x0 = ", x0)
        next_state, curr_control = nplayer_hierarchy_navigation(x0; verbose=false)
        
        @test length(next_state) == 3
        @test length(curr_control) == 3
        @test all(isfinite.(vcat(next_state...)))
        @test all(isfinite.(vcat(curr_control...)))
        
        println("  Next states: ", next_state)
        println("  Controls: ", curr_control)
    end
end
