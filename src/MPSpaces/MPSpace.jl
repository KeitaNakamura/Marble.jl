struct MPSpace{dim, FT <: ShapeFunction{dim}, GT <: AbstractGrid{dim}, VT <: ShapeValue{dim}}
    F::FT
    grid::GT
    dofmap::DofMap{dim}
    dofindices::Vector{Vector{Int}}
    gridindices::Vector{Vector{CartesianIndex{dim}}}
    activeindices::Vector{CartesianIndex{dim}}
    freedofs::Vector{Int} # flat dofs
    bounddofs::Vector{Int}
    nearsurface::BitVector
    Nᵢ::PointState{VT}
end

function MPSpace(::Type{T}, F::ShapeFunction{dim}, grid::AbstractGrid{dim}, npoints::Int) where {dim, T <: Real}
    dofmap = DofMap(size(grid))
    dofindices = [Int[] for _ in 1:npoints]
    gridindices = [CartesianIndex{dim}[] for _ in 1:npoints]
    activeindices = CartesianIndex{dim}[]
    freedofs = Int[]
    bounddofs = Int[]
    nearsurface = falses(npoints)
    Nᵢ = pointstate([construct(T, F) for _ in 1:npoints])
    MPSpace(F, grid, dofmap, dofindices, gridindices, activeindices, freedofs, bounddofs, nearsurface, Nᵢ)
end

MPSpace(F::ShapeFunction, grid::AbstractGrid, npoints::Int) = MPSpace(Float64, F, grid, npoints)

value_gradient_type(::Type{T}, ::Val{dim}) where {T <: Real, dim} = ScalVec{dim, T}
value_gradient_type(::Type{Vec{dim, T}}, ::Val{dim}) where {T, dim} = VecTensor{dim, T, dim^2}

function reinit_dofmap!(space::MPSpace{dim}, coordinates; exclude = nothing, point_radius::Real) where {dim}
    dofindices = space.dofindices
    gridindices = space.gridindices
    @assert length(coordinates) == length(dofindices) == length(gridindices)

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

    # Initialize dof indices by updated DofMap
    @inbounds for (i, x) in enumerate(coordinates)
        allinds = neighboring_nodes(grid, x, point_radius)
        DofHelpers.map!(dofmap, dofindices[i], allinds)
        allactive = DofHelpers.filter!(dofmap, gridindices[i], allinds)
        space.nearsurface[i] = !allactive
    end

    ## active grid indices
    DofHelpers.filter!(dofmap, space.activeindices, CartesianIndices(dofmap))

    ## freedofs (used in dirichlet boundary conditions)
    # TODO: modify for scalar field: need to create freedofs for scalar field?
    empty!(space.freedofs)
    @inbounds for i in CartesianIndices(dofmap)
        I = dofmap(i; dof = dim)
        I === nothing && continue
        if onbound(dofmap, i)
            for d in 1:dim
                if !onbound(size(dofmap, d), i[d])
                    push!(space.freedofs, I[d])
                end
            end
        else
            append!(space.freedofs, I)
        end
    end

    ## bounddofs (!!NOT!! flat)
    empty!(space.bounddofs)
    @inbounds for i in CartesianIndices(dofmap)
        I = dofmap(i)
        I === nothing && continue
        if onbound(dofmap, i)
            push!(space.bounddofs, I)
        end
    end

    space
end

function reinit_shapevalue!(space::MPSpace, coordinates)
    @inbounds Threads.@threads for p in 1:npoints(space)
        inds = gridindices(space, p)
        reinit!(space.Nᵢ[p], space.grid, inds, coordinates[p])
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
    gridstate(T, space.dofmap, space.dofindices)
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
        return gridstate(view(A, space.activeindices), space.dofmap, space.dofindices)
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
