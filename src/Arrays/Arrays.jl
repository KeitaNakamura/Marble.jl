module Arrays

using Poingr.DofHelpers
using SparseArrays
import SparseArrays: sparse, sparse!, nonzeros, nnz

using Base: @_propagate_inbounds_meta

export
# FillArray
    FillArray,
    Ones,
    Zeros,
# ScalarMatrix
    ScalarMatrix,
# SparseMatrixCOO
    SparseMatrixCOO,
    SparseMatrixCSC,
    sparse,
    sparse!,
# SparseArray
    SparseArray,
    nonzeros,
    nzindices,
    nnz,
# List
    List,
    ListGroup

include("FillArray.jl")
include("ScalarMatrix.jl")
include("SparseMatrixCOO.jl")
include("SparseArray.jl")
include("List.jl")
include("DashedUnitRange.jl")

end
