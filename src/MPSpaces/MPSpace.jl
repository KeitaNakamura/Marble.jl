struct MPSpace{dim, FT <: ShapeFunction{dim}, GT <: AbstractGrid{dim}, VT <: ShapeValue{dim}}
    F::FT
    grid::GT
    dofmap::DofMap{dim}
    dofindices::PointToDofIndices
    gridindices::PointToGridIndices{dim}
    activeindices::Vector{CartesianIndex{dim}}
    freedofs::Vector{Int} # flat dofs
    bounddofs::Vector{Int}
    nearsurface::BitVector
    Nᵢ::PointState{VT}
    pointsincell::Array{Vector{Int}, dim}
    colors::ColoredBlocks{dim}
end

function chunk_ranges(total::Int, nchunks::Int)
    splits = [round(Int, s) for s in range(0, stop=total, length=nchunks+1)]
    [splits[i]+1:splits[i+1] for i in 1:nchunks]
end

function MPSpace(::Type{T}, F::ShapeFunction{dim}, grid::AbstractGrid{dim}, npoints::Int) where {dim, T <: Real}
    dofmap = DofMap(size(grid))

    construct_dofindices(n) = [Int[] for _ in 1:n]

    dofindices = construct_dofindices(npoints)
    gridindices = [CartesianIndex{dim}[] for _ in 1:npoints]

    activeindices = CartesianIndex{dim}[]
    freedofs = Int[]
    bounddofs = Int[]
    nearsurface = falses(npoints)
    Nᵢ = pointstate([construct(T, F) for _ in 1:npoints])

    pointsincell = [Int[] for i in CartesianIndices(size(grid) .- 1)]
    colors = coloringcells(grid, 6)

    MPSpace(F, grid, dofmap, dofindices, gridindices, activeindices, freedofs, bounddofs, nearsurface, Nᵢ, pointsincell, colors)
end

MPSpace(F::ShapeFunction, grid::AbstractGrid, npoints::Int) = MPSpace(Float64, F, grid, npoints)

value_gradient_type(::Type{T}, ::Val{dim}) where {T <: Real, dim} = ScalVec{dim, T}
value_gradient_type(::Type{Vec{dim, T}}, ::Val{dim}) where {T, dim} = VecTensor{dim, T, dim^2}

function _reinit!(space, coordinates, exclude, point_radius)
    dofmap = space.dofmap
    dofindices = space.dofindices
    gridindices = space.gridindices
    grid = space.grid

    @assert length(coordinates) == length(dofindices)

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
        @inbounds for i in eachindex(grid, dofmap)
            if dofmap[i] == true
                xi = grid[i]
                exclude(xi) && (dofmap[i] = false)
            end
        end
        # surrounding nodes are activated
        for x in coordinates
            inds = neighboring_nodes(grid, x, 1)
            @inbounds dofmap[inds] .= true
        end
    end

    ## renumering dofs
    count!(dofmap)

    # Reinitialize shape values and dof indices by updated DofMap
    empty!.(space.pointsincell)
    @inbounds for i in 1:length(coordinates)
        allinds = neighboring_nodes(grid, coordinates[i], point_radius)
        DofHelpers.map!(dofmap, dofindices[i], allinds)
        allactive = DofHelpers.filter!(dofmap, gridindices[i], allinds)
        space.nearsurface[i] = !allactive
        push!(space.pointsincell[whichcell(grid, coordinates[i])], i)
    end
end

function reinit!(space::MPSpace{dim}, coordinates; exclude = nothing) where {dim}
    point_radius = ShapeFunctions.support_length(space.F)
    _reinit!(space, coordinates, exclude, point_radius)

    @assert length(space.gridindices) == length(coordinates)

    ## active grid indices
    DofHelpers.filter!(space.dofmap, space.activeindices, CartesianIndices(space.dofmap))

    ## freedofs (used in dirichlet boundary conditions)
    # TODO: modify for scalar field: need to create freedofs for scalar field?
    empty!(space.freedofs)
    @inbounds for i in CartesianIndices(space.dofmap)
        I = space.dofmap(i; dof = dim)
        I === nothing && continue
        if onbound(space.dofmap, i)
            for d in 1:dim
                if !onbound(size(space.dofmap, d), i[d])
                    push!(space.freedofs, I[d])
                end
            end
        else
            append!(space.freedofs, I)
        end
    end

    ## bounddofs (!!NOT!! flat)
    empty!(space.bounddofs)
    @inbounds for i in CartesianIndices(space.dofmap)
        I = space.dofmap(i)
        I === nothing && continue
        if onbound(space.dofmap, i)
            push!(space.bounddofs, I)
        end
    end

    Threads.@threads for p in eachindex(coordinates)
        inds = gridindices(space, p)
        reinit!(space.Nᵢ[p], space.grid, inds, coordinates[p])
    end

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
    DofHelpers.map(space.dofmap, space.gridindices[p]; dof)
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
gridsize(space::MPSpace) = size(space.grid)

function gridstate(space::MPSpace, T)
    gridstate(T, space.dofmap, space.dofindices, space.gridindices, space.pointsincell, space.colors)
end

function gridstate_matrix(space::MPSpace, T)
    gridstate_matrix(T, space.dofindices, space.freedofs)
end

function pointstate(space::MPSpace, T)
    pointstate(T, npoints(space))
end

function construct(name::Symbol, space::MPSpace)
    name == :shape_value        && return space.Nᵢ
    name == :shape_vector_value && return lazy(vec, space.Nᵢ)
    if name == :bound_normal_vector
        A = BoundNormalArray(Float64, gridsize(space)...)
        return GridState(SparseArray(view(A, space.activeindices), space.dofmap), space.dofindices, space.gridindices, space.pointsincell, space.colors)
    end
    if name == :grid_coordinates
        return GridStateCollection(view(space.grid, space.activeindices), space.dofindices)
    end
    if eltype(space.Nᵢ) <: WLSValue
        if name == :weight_value
            return lazy(ShapeFunctions.weight_value, space.Nᵢ)
        end
        if name == :moment_matrix_inverse
            return lazy(ShapeFunctions.moment_matrix_inverse, space.Nᵢ)
        end
    end
    throw(ArgumentError("$name in $(space.F) is not supported"))
end

function dirichlet!(vᵢ::GridState{dim, Vec{dim, T}}, space::MPSpace{dim}) where {dim, T}
    V = reinterpret(T, nonzeros(vᵢ))
    fixeddofs = setdiff(1:ndofs(space, dof = dim), space.freedofs)
    V[fixeddofs] .= zero(T)
    vᵢ
end
