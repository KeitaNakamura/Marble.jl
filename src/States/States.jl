module States

using Jams.Arrays
using Jams.Collections
using Jams.DofHelpers
using Jams.Grids
using Base: @_propagate_inbounds_meta
using Base.Broadcast: broadcasted

import SparseArrays: nonzeros, nnz, sparse!
import Jams.DofHelpers: indices

export
# PointState
    PointState,
    pointstate,
    ←,
    generate_pointstates,
# GridState
    GridState,
    gridstate,
    nonzeros, # from SparseArrays
    nnz,      # from SparseArrays
    zeros!,
# GridStateMatrix
    GridStateMatrix,
    gridstate_matrix,
    sparse!,
# GridDiagonal
    GridDiagonal,
# GridStateCollection
    GridStateCollection,
# PointToGridOperation
    ∑ₚ,
# GridToPointOperation
    ∑ᵢ

include("PointState.jl")
include("GridState.jl")
include("GridStateMatrix.jl")
include("GridStateOperation.jl")
include("GridStateCollection.jl")
include("PointToGridOperation.jl")
include("GridToPointOperation.jl")

end
