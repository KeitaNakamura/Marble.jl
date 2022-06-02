###############
# Grid states #
###############

struct DefaultGridState{dim, T, L, LL}
    m::T
    v::Vec{dim, T}
    v_n::Vec{dim, T}
    poly_coef::Vec{L, T}
    poly_mat::Mat{L, L, T, LL}
end

default_gridstate_type(::Interpolation, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultGridState{dim, T, dim+1, (dim+1)*(dim+1)}

################
# Point states #
################

default_pointstate_type(::Nothing, ::Val{dim}, ::Val{T}) where {dim, T} =
    @NamedTuple{x::Vec{dim, T}, V::T, r::Vec{dim, T}, index::Int}

struct DefaultPointState{dim, T, L, dim_L}
    m::T
    V::T
    x::Vec{dim, T}
    v::Vec{dim, T}
    r::Vec{dim, T}
    b::Vec{dim, T}
    σ::SymmetricSecondOrderTensor{3, T, 6}
    ϵ::SymmetricSecondOrderTensor{3, T, 6}
    ∇v::SecondOrderTensor{3, T, 9}
    C::Mat{dim, L, T, dim_L}
    index::Int
end

default_pointstate_type(::Interpolation, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultPointState{dim, T, dim, dim*dim}
default_pointstate_type(::LinearWLS, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultPointState{dim, T, dim+1, dim*(dim+1)}
default_pointstate_type(::BilinearWLS, ::Val{2}, ::Val{T}) where {T} = DefaultPointState{2, T, 4, 8}

function generate_pointstate(indomain, Point::Type, grid::Grid{T, dim}; n::Int = 2) where {dim, T}
    h = gridsteps(grid) ./ n # length per particle
    allpoints = Grid(@. LinRange(
        first($gridaxes(grid)) + h/2,
        last($gridaxes(grid))  - h/2,
        n * ($size(grid) - 1)
    ))

    npoints = count(x -> indomain(x...), allpoints)
    pointstate = StructVector{Point}(undef, npoints)
    fillzero!(pointstate)

    if :x in propertynames(pointstate)
        cnt = 0
        for x in allpoints
            if indomain(x...)
                @inbounds pointstate.x[cnt+=1] = x
            end
        end
    end
    if :V in propertynames(pointstate)
        V = prod(h)
        if dim == 2 && grid.coordinate_system isa Axisymmetric
            @. pointstate.V = getindex(pointstate.x, 1) * V
        else
            @. pointstate.V = V
        end
    end
    if :r in propertynames(pointstate)
        pointstate.r .= Vec(h) / 2
    end
    if :index in propertynames(pointstate)
        pointstate.index .= 1:npoints
    end

    reorder_pointstate!(pointstate, pointsinblock(grid, pointstate.x))
    pointstate
end

function generate_pointstate(indomain, grid::Grid{T, dim}; kwargs...) where {dim, T}
    Point = default_pointstate_type(grid.interpolation, Val(dim), Val(T))
    generate_pointstate(indomain, Point, grid; kwargs...)
end

function points_outside_domain(xₚ::AbstractVector, grid::Grid)
    findall(xₚ) do x
        @inbounds begin
            !(grid[begin][1] ≤ x[1] ≤ grid[end][1] &&
              grid[begin][2] ≤ x[2] ≤ grid[end][2])
        end
    end
end
