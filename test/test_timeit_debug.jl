using Test
using TimerOutputs: TimerOutput, TimerOutputs, ncalls

@testset "@timeit_debug macro" begin
    @testset "TIMING_ENABLED flag exists and defaults to false" begin
        @test MixedHierarchyGames.TIMING_ENABLED isa Threads.Atomic{Bool}
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

    @testset "is_timing_enabled returns current state" begin
        MixedHierarchyGames.disable_timing!()
        @test MixedHierarchyGames.is_timing_enabled() == false

        MixedHierarchyGames.enable_timing!()
        @test MixedHierarchyGames.is_timing_enabled() == true

        MixedHierarchyGames.disable_timing!()
    end

    @testset "with_timing enables timing and restores state" begin
        MixedHierarchyGames.disable_timing!()
        @test MixedHierarchyGames.is_timing_enabled() == false

        executed = Ref(false)
        MixedHierarchyGames.with_timing() do
            @test MixedHierarchyGames.is_timing_enabled() == true
            executed[] = true
        end
        @test executed[] == true
        @test MixedHierarchyGames.is_timing_enabled() == false
    end

    @testset "with_timing restores previous state on exception" begin
        MixedHierarchyGames.disable_timing!()
        try
            MixedHierarchyGames.with_timing() do
                error("test error")
            end
        catch
        end
        @test MixedHierarchyGames.is_timing_enabled() == false
    end

    @testset "with_timing restores enabled state if already enabled" begin
        MixedHierarchyGames.enable_timing!()
        MixedHierarchyGames.with_timing() do
            @test MixedHierarchyGames.is_timing_enabled() == true
        end
        @test MixedHierarchyGames.is_timing_enabled() == true

        MixedHierarchyGames.disable_timing!()
    end

    @testset "No timing data collected when disabled" begin
        MixedHierarchyGames.disable_timing!()
        to = TimerOutput()

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
