using Test
using TimerOutputs: TimerOutput, TimerOutputs, ncalls

@testset "@timeit_debug macro" begin
    @testset "TIMING_ENABLED flag exists and defaults to false" begin
        @test MixedHierarchyGames.TIMING_ENABLED isa Ref{Bool}
        @test MixedHierarchyGames.TIMING_ENABLED[] == false
    end

    @testset "enable_timing! and disable_timing! control the flag" begin
        # Should start disabled
        @test MixedHierarchyGames.TIMING_ENABLED[] == false

        MixedHierarchyGames.enable_timing!()
        @test MixedHierarchyGames.TIMING_ENABLED[] == true

        MixedHierarchyGames.disable_timing!()
        @test MixedHierarchyGames.TIMING_ENABLED[] == false
    end

    @testset "No timing data collected when disabled" begin
        MixedHierarchyGames.disable_timing!()
        to = TimerOutput()

        # Call a function that uses @timeit_debug internally
        # We use a simple test: call the macro expansion directly
        MixedHierarchyGames.@timeit_debug to "test section" begin
            x = 1 + 1
        end

        # TimerOutput should have no recorded sections
        @test !haskey(to, "test section")
    end

    @testset "Timing data collected when enabled" begin
        MixedHierarchyGames.enable_timing!()
        to = TimerOutput()

        MixedHierarchyGames.@timeit_debug to "test section" begin
            x = 1 + 1
        end

        # TimerOutput should have recorded the section
        @test haskey(to, "test section")
        @test ncalls(to["test section"]) >= 1

        MixedHierarchyGames.disable_timing!()
    end

    @testset "Code block executes regardless of timing state" begin
        # When disabled
        MixedHierarchyGames.disable_timing!()
        result_disabled = Ref(0)
        to = TimerOutput()
        MixedHierarchyGames.@timeit_debug to "disabled block" begin
            result_disabled[] = 42
        end
        @test result_disabled[] == 42

        # When enabled
        MixedHierarchyGames.enable_timing!()
        result_enabled = Ref(0)
        MixedHierarchyGames.@timeit_debug to "enabled block" begin
            result_enabled[] = 99
        end
        @test result_enabled[] == 99

        MixedHierarchyGames.disable_timing!()
    end

    @testset "Return value is preserved" begin
        MixedHierarchyGames.disable_timing!()
        to = TimerOutput()
        val = MixedHierarchyGames.@timeit_debug to "return test" begin
            42
        end
        @test val == 42

        MixedHierarchyGames.enable_timing!()
        val2 = MixedHierarchyGames.@timeit_debug to "return test enabled" begin
            99
        end
        @test val2 == 99

        MixedHierarchyGames.disable_timing!()
    end

    @testset "Nested @timeit_debug sections work when enabled" begin
        MixedHierarchyGames.enable_timing!()
        to = TimerOutput()

        MixedHierarchyGames.@timeit_debug to "outer" begin
            MixedHierarchyGames.@timeit_debug to "inner" begin
                x = 1 + 1
            end
        end

        @test haskey(to, "outer")
        @test haskey(to["outer"], "inner")

        MixedHierarchyGames.disable_timing!()
    end

    @testset "reset_timer! clears timing data" begin
        MixedHierarchyGames.enable_timing!()
        to = TimerOutput()

        MixedHierarchyGames.@timeit_debug to "before reset" begin
            x = 1 + 1
        end
        @test haskey(to, "before reset")

        TimerOutputs.reset_timer!(to)
        @test !haskey(to, "before reset")

        MixedHierarchyGames.disable_timing!()
    end
end
