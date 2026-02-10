using Documenter
using MixedHierarchyGames

makedocs(;
    modules=[MixedHierarchyGames],
    sitename="MixedHierarchyGames.jl",
    repo=Documenter.Remotes.GitHub("CLeARoboticsLab", "MixedHierarchyGames.jl"),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
    # Docstrings use array-index notation like gs[i](z) which Documenter
    # parses as markdown links. Downgrade to warnings instead of errors.
    warnonly=[:cross_references],
)

deploydocs(;
    repo="github.com/CLeARoboticsLab/MixedHierarchyGames.jl.git",
    devbranch="main",
    push_preview=true,
)
