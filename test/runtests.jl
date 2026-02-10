using Test
using MixedHierarchyGames

# Include shared testing utilities
include("testing_utils.jl")

# Include test tier configuration
include("test_tiers.jl")

# Determine which tests to run based on FAST_TESTS_ONLY environment variable
fast_only = get(ENV, "FAST_TESTS_ONLY", "false") == "true"
test_files = get_test_files(fast_only)

if fast_only
    @info "Running FAST tests only (set FAST_TESTS_ONLY=false or unset to run all)"
else
    @info "Running ALL tests (fast + slow)"
end

@testset "MixedHierarchyGames.jl" begin
    # Test tier configuration self-test
    include("test_test_tiers.jl")

    for file in test_files
        include(file)
    end
end
