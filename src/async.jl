struct Time <: Real
    T::Rational{Int}
    tϵ::Float64
end
checktimecoef(a::Time, b::Time) = @assert a.tϵ == b.tϵ
Base.convert(::Type{T}, t::Time) where {T <: Real} = convert(T, t.tϵ * t.T)
Base.convert(::Type{Time}, t::Time) where {T <: Real} = t
Base.:+(a::Time, b::Time) = (checktimecoef(a, b); Time(a.T + b.T, a.tϵ))
Base.:-(a::Time, b::Time) = (checktimecoef(a, b); Time(a.T - b.T, a.tϵ))
Base.promote_type(::Type{Time}, ::Type{<: Real}) = Float64
Base.promote_type(::Type{<: Real}, ::Type{Time}) = Float64
Base.show(io::IO, time::Time) = print(io, "t = ", time.tϵ * time.T, " (", time.tϵ, " * ", time.T, ")")

struct BlockedVector{T, V <: AbstractVector{T}} <: AbstractVector{T}
    data::Vector{V}
    axis::BlockArrays.BlockedUnitRange{Vector{Int}}
end
BlockedVector(data::Vector) = BlockedVector(data, BlockArrays.blockedrange(length.(data)))
Base.size(x::BlockedVector) = (sum(length, x.data),)
Base.axes(x::BlockedVector) = (x.axis,)
@inline function Base.getindex(x::BlockedVector, i::Int)
    @boundscheck checkbounds(x, i)
    index = BlockArrays.findblockindex(axes(x, 1), i)
    @inbounds x.data[index.I[1]][index.α[1]]
end
@inline function Base.setindex!(x::BlockedVector, v, i::Int)
    @boundscheck checkbounds(x, i)
    index = BlockArrays.findblockindex(axes(x, 1), i)
    @inbounds x.data[index.I[1]][index.α[1]] = v
    x
end

blockedvector(data::Vector) = BlockedVector(data)
_map_getproperty(::Val{name}, data) where {name} = map(x -> getproperty(x, name), data)
@generated function blockedvector(data::Vector{<: StructVector{T, <: NamedTuple{names}}}) where {T, names}
    exps = [:(BlockedVector(_map_getproperty($(Val(name)), data), axis)) for name in names]
    quote
        axis = BlockArrays.blockedrange(length.(data))
        StructVector{T}(($(exps...),))
    end
end

mutable struct Block{PointState <: StructVector}
    pointstate::PointState
    buffer::PointState
    T::Rational{Int}
    T_buffer::Rational{Int}
    T_local::Rational{Int}
    dT::Rational{Int}
end

mutable struct Scheduler{PointState, dim}
    blocks::Array{Block{PointState}, dim}
    gridsize::NTuple{dim, Int}
    time::Time
end

gridsize(sch::Scheduler) = sch.gridsize
currenttime(sch::Scheduler) = sch.time

function issynced(sch::Scheduler)
    blocks = sch.blocks
    candidates = Iterators.filter(blk -> !isempty(blk.pointstate), blocks)
    Ts = map(blk -> blk.T, candidates)
    all(==(Ts[1]), Ts)
end

function gather_pointstate(sch::Scheduler)
    @assert issynced(sch)
    blockedvector(vec(map(blk -> blk.pointstate, sch.blocks)))
end

function Scheduler(grid::Grid, pointstate::PointState, tϵ::Real) where {PointState <: StructVector}
    blocks = map(pointsinblock(grid, pointstate.x)) do pointindices
        ps = pointstate[pointindices]
        buf = copy(ps)
        Block{PointState}(ps, buf, 0, 0, 0, 1)
    end
    Scheduler(blocks, size(grid), Time(0, tϵ))
end

function updatetimestep!(calculate_timestep::Function, sch::Scheduler, grid::Grid; exclude = nothing)
    time = currenttime(sch)
    blocks = sch.blocks

    nearsurface = falses(size(blocks))
    if exclude !== nothing
        gridmask = falses(size(grid))
        for color in coloringblocks(size(grid))
            @inbounds Threads.@threads for I in color
                xₚ = blocks[I].pointstate.x
                for x in xₚ
                    inds = neighboring_nodes(grid, x, 1)
                    gridmask[inds] .= true
                end
            end
        end
        Threads.@threads for I in eachindex(blocks)
            block = blocks[I]
            xₚ = block.pointstate.x
            isnearsurface = false
            for x in xₚ
                @inbounds for i in neighboring_nodes(grid, x)
                    if !gridmask[i] && exclude(grid[i])
                        isnearsurface = true
                        break
                    end
                end
                isnearsurface && break
            end
            if isnearsurface
                nearsurface[I] = true
            end
        end
    end

    # for non-empty blocks
    dTmin = 1//0
    dTmax = 0//1
    Threads.@threads for block in blocks
        (mod(time.T, block.dT) == 0 && !isempty(block.pointstate)) || continue
        limit = minimum(calculate_timestep, block.pointstate) / time.tϵ
        while limit < block.dT
            block.dT /= 2
        end
        while limit ≥ 2*block.dT && mod(time.T, 2*block.dT) == 0
            block.dT *= 2
        end
        dTmin = min(dTmin, block.dT)
        dTmax = max(dTmax, block.dT)
    end

    # for empty blocks
    @inbounds Threads.@threads for block in blocks
        (mod(time.T, block.dT) == 0 && isempty(block.pointstate)) || continue
        while dTmax < block.dT
            block.dT /= 2
        end
        while dTmax ≥ 2*block.dT && mod(time.T, 2*block.dT) == 0
            block.dT *= 2
        end
    end

    # for nearsurface blocks
    @inbounds Threads.@threads for I in eachindex(blocks)
        block = blocks[I]
        if nearsurface[I]
            block.dT = dTmin
        end
    end

    sch
end

function advance!(microstep::Function, sch::Scheduler, grid::Grid, dtime::Time)
    time = currenttime(sch)

    dT = dtime.T
    blocks = sch.blocks

    mask_equal = falses(size(blocks))
    mask_larger = falses(size(blocks))
    mask_smaller = falses(size(blocks))
    @inbounds for I in CartesianIndices(blocks)
        block = blocks[I]
        if block.T == time.T && block.dT == dT
            mask_equal[I] = true
            for J in neighboring_blocks(grid, I, 1)
                block_nearby = blocks[J]
                if block_nearby.dT > block.dT
                    mask_larger[J] = true
                    @assert block.T == block_nearby.T_buffer
                end
                if block_nearby.dT < block.dT
                    mask_smaller[J] = true
                    @assert block.T == block_nearby.T
                end
            end
        end
    end
    @assert !any(mask_equal .& mask_larger)
    @assert !any(mask_equal .& mask_smaller)
    @assert !any(mask_larger .& mask_smaller)

    @inbounds Threads.@threads for I in eachindex(blocks)
        block = blocks[I]
        if mask_equal[I] || mask_smaller[I]
            copy!(block.buffer, block.pointstate)
            block.T_buffer = block.T
        end
    end

    blocks_equal   = [ blocks[i].pointstate for i in eachindex(blocks) if mask_equal[i]   ]
    blocks_larger  = [ blocks[i].buffer     for i in eachindex(blocks) if mask_larger[i]  ]
    blocks_smaller = [ blocks[i].buffer     for i in eachindex(blocks) if mask_smaller[i] ]
    pointstate = blockedvector(vcat(blocks_equal, blocks_larger, blocks_smaller))

    microstep(pointstate, dtime)

    # check particles moving to nearby blocks
    @inbounds for I in CartesianIndices(blocks)
        block = blocks[I]
        if mask_equal[I]
            pstate = block.pointstate
            block.T += dT # advance block for pointstate
        elseif mask_larger[I]
            pstate = block.buffer
            block.T_buffer += dT # advance block for buffer
        elseif mask_smaller[I]
            pstate = block.buffer
        else
            continue
        end
        p = 1
        while p ≤ length(pstate)
            b = whichblock(grid, pstate.x[p]) # recompute which block
            if b === nothing
                deleatat!(pstate, p)
                continue
            end
            if b != I # move to nearby block
                # particles can freely move across the 'equal' and 'larger' blocks
                # if the particles moved to other blocks, they should be removed from original block.
                if mask_equal[b]
                    push!(blocks[b].pointstate, popat!(pstate, p))
                    continue
                elseif mask_larger[b]
                    push!(blocks[b].buffer, popat!(pstate, p))
                    continue
                else
                    deleteat!(pstate, p)
                    continue
                end
            end
            p += 1
        end
        if mask_larger[I]
            if block.T_buffer == block.T
                empty!(block.buffer)
            end
        end
    end
end

function asyncrun!(microstep::Function, sch::Scheduler, grid::Grid)
    time = currenttime(sch)
    dTs = sort(unique(map(block -> block.dT, sch.blocks)), rev = true)
    for dT in dTs
        if mod(time.T, dT) == 0
            @show dT
            advance!(microstep, sch, grid, Time(dT, time.tϵ))
        end
    end
    dTmin = dTs[end]
    dT = dTmin - mod(time.T, dTmin)
    dtime = Time(dT, time.tϵ)
    sch.time += dtime
    dtime
end
