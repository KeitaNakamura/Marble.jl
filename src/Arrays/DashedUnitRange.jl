struct DashedUnitRange{T} <: AbstractVector{UnitRange{T}}
    parent::UnitRange{T}
    on::Int
    off::Int
end

Base.parent(dashed::DashedUnitRange) = dashed.parent

Base.size(dashed::DashedUnitRange) = (length(dashedstartinds(dashed)),)

function dashedstartinds(dashed)
    p = parent(dashed)
    firstindex(p):dashed.on+dashed.off:lastindex(p)
end

function Base.getindex(dashed::DashedUnitRange, i::Int)
    startinds = dashedstartinds(dashed)
    @boundscheck checkbounds(startinds, i)
    @inbounds begin
        start = startinds[i]
        p = parent(dashed)
        p[start:min(start+dashed.on-1, lastindex(p))]
    end
end
