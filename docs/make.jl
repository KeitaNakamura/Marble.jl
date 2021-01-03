using Documenter
using Jams

# Setup for doctests in docstrings
DocMeta.setdocmeta!(Jams, :DocTestSetup, recursive = true,
    quote
        using Jams
    end
)

makedocs(;
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Jams],
    sitename = "Jams.jl",
    pages=[
        "Home" => "index.md",
        "Jams.DofHelpers" => "DofHelpers.md",
        "Jams.Arrays" => "Arrays.md",
        "Jams.Grids" => "Grids.md",
        "Jams.ShapeFunctions" => "ShapeFunctions.md",
        "Jams.MPSpaces" => "MPSpaces.md",
    ],
    doctest = true, # :fix
)

deploydocs(
    repo = "github.com/Nakamura-Lab/Jams.jl.git",
)
