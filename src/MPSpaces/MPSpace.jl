struct MPSpace{dim, FT <: ShapeFunction{dim}, GT <: AbstractGrid{dim}, VT <: ShapeValue{dim}}
    F::FT
    grid::GT
    dofmap::DofMap{dim}
    dofindices::Vector{Vector{Int}}
    dofindices_dim::Vector{Vector{Int}}
    gridindices::Vector{Vector{CartesianIndex{dim}}}
    Nᵢ::PointState{VT}
end

function MPSpace(::Type{T}, F::ShapeFunction{dim}, grid::AbstractGrid{dim}, npoints::Int) where {dim, T <: Real}
    dofmap = DofMap(size(grid))
    dofindices = [Int[] for _ in 1:npoints]
    dofindices_dim = [Int[] for _ in 1:npoints]
    gridindices = [CartesianIndex{dim}[] for _ in 1:npoints]
    Nᵢ = pointstate([construct(T, F) for _ in 1:npoints])
    MPSpace(F, grid, dofmap, dofindices, dofindices_dim, gridindices, Nᵢ)
end

MPSpace(F::ShapeFunction, grid::AbstractGrid, npoints::Int) = MPSpace(Float64, F, grid, npoints)

value_gradient_type(::Type{T}, ::Val{dim}) where {T <: Real, dim} = ScalVec{dim, T}
value_gradient_type(::Type{Vec{dim, T}}, ::Val{dim}) where {T, dim} = VecTensor{dim, T, dim^2}

function reinit_dofmap!(space::MPSpace{dim}, coordinates; exclude = nothing, point_radius::Real) where {dim}
    dofindices = space.dofindices
    dofindices_dim = space.dofindices_dim
    gridindices = space.gridindices
    @assert length(coordinates) == length(dofindices) == length(dofindices_dim) == length(gridindices)

    grid = space.grid
    dofmap = space.dofmap

    # Reinitialize dofmap

    ## reset
    dofmap .= false

    ## activate grid indices and store them.
    for x in coordinates
        inds = neighboring_nodes(grid, x, point_radius)
        @inbounds dofmap[inds] .= true
    end

    ## exclude grid nodes if a function is given
    if exclude !== nothing
        # if exclude(xi) is true, then make it false
        @inbounds for i in eachindex(grid)
            xi = grid[i]
            exclude(xi) && (dofmap[i] = false)
        end
        # surrounding nodes are activated
        for x in coordinates
            inds = neighboring_nodes(grid, x, 1)
            @inbounds dofmap[inds] .= true
        end
    end

    ## renumering dofs
    count!(dofmap)

    # Reinitialize interpolations by updated mask
    @inbounds for (i, x) in enumerate(coordinates)
        allinds = neighboring_nodes(grid, x, point_radius)
        DofHelpers.map!(dofmap, dofindices[i], allinds)
        DofHelpers.map!(dofmap, dofindices_dim[i], allinds; dof = dim)
        DofHelpers.filter!(dofmap, gridindices[i], allinds)
    end

    space
end

function reinit_shapevalue!(space::MPSpace, coordinates)
    @inbounds for (j, (x, N)) in enumerate(zip(coordinates, space.Nᵢ))
        inds = gridindices(space, j)
        reinit!(N, space.grid, inds, x)
    end
    space
end

function reinit!(space::MPSpace, coordinates; exclude = nothing)
    point_radius = ShapeFunctions.support_length(space.F)
    reinit_dofmap!(space, coordinates; point_radius, exclude)
    reinit_shapevalue!(space, coordinates)
    space
end

"""
    dofindices(::MPSpace, p::Int; [dof])

Return dof indices at point index `p`.
Use [`reinit!(::MPSpace, ::AbstractArray{<: Vec})`](@ref) in advance.
"""
function dofindices(space::MPSpace{dim}, p::Int; dof::Int = 1) where {dim}
    @_propagate_inbounds_meta
    dof == 1   && return space.dofindices[p]
    dof == dim && return space.dofindices_dim[p]
    DofHelpers.map(space.dofmap, space.gridindices[p])
end

"""
    gridindices(::MPSpace, p::Int)

Return grid indices at point index `p`.
Use [`reinit!(::MPSpace, ::AbstractArray{<: Vec})`](@ref) in advance.
"""
gridindices(space::MPSpace, p::Int) = (@_propagate_inbounds_meta; space.gridindices[p])

"""
    ndofs(::MPSpace; [dof])

Return total number of dofs.
"""
ndofs(space::MPSpace; dof::Int = 1) = ndofs(space.dofmap; dof)

npoints(space::MPSpace) = length(space.dofindices)

function gridstate(space::MPSpace, T)
    gridstate(T, space.dofmap, space.dofindices)
end

function gridstate_matrix(space::MPSpace, T)
    gridstate_matrix(T, space.dofindices)
end

function pointstate(space::MPSpace, T)
    pointstate(T, npoints(space))
end

function function_space(space::MPSpace, name::Symbol)
    name == :shape_function        && return space.Nᵢ
    name == :shape_function_vector && return lazy(vec, space.Nᵢ)
    throw(ArgumentError("not supported function space"))
end

function dirichlet!(vᵢ::GridState{dim, Vec{dim, T}}, space::MPSpace{dim}) where {dim, T}
    grid = space.grid
    dofmap = space.dofmap
    V = reinterpret(T, nonzeros(vᵢ))
    for boundset in values(getboundsets(grid))
        @inbounds for bound in boundset
            I = dofmap(bound.index; dof = dim)
            I === nothing && continue
            V[I[bound.component]] = zero(T)
        end
    end
    vᵢ
end