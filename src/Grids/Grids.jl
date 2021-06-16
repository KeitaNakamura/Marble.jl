module Grids

using Reexport
using Coordinates
using Poingr.Arrays: DashedUnitRange
@reexport using Poingr.TensorValues

using Base: @_inline_meta, @_propagate_inbounds_meta
using Base.Cartesian: @ntuple, @nall

export
# AbstractGrid
    AbstractGrid,
    gridaxes,
    gridorigin,
    gridsteps,
# Grid
    Grid,
# neighboring
    neighboring_nodes,
    neighboring_cells,
    whichcell,
# coloring
    coloringcells

include("AbstractGrid.jl")
include("Grid.jl")
include("neighboring.jl")
include("coloring.jl")

end # module
