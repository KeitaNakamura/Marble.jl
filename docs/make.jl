using Documenter
using Metale

# Setup for doctests in docstrings
DocMeta.setdocmeta!(Metale, :DocTestSetup, recursive = true,
    quote
        using Metale
    end
)

makedocs(;
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Metale],
    sitename = "Metale.jl",
    pages=[
        "Home" => "index.md",
        "Grid" => "grid.md",
        "Interpolations" => "interpolations.md",
        "Contact mechanics" => "contact_mechanics.md",
        "VTK outputs" => "VTK_outputs.md",
    ],
    doctest = true, # :fix
)

deploydocs(
    repo = "github.com/KeitaNakamura/Metale.jl.git",
    devbranch = "main",
)
