@testset "SpPattern" begin
    @test @inferred(Metale.SpPattern((5,5))) == falses(5,5)
    @test @inferred(Metale.SpPattern(5,5)) == falses(5,5)

    spat = Metale.SpPattern(5,5)

    # getindex/setindex!
    inds = fill(-1, size(spat))
    true_inds = 1:2:length(spat)
    for (i, I) in enumerate(true_inds)
        spat[I] = true
        @test spat[I] == true
        inds[I] = i # for latter test
    end
    for i in setdiff(1:length(spat), true_inds)
        @test spat[i] == false
    end

    # reinit!
    @test Metale.reinit!(spat) == count(spat)
    @test spat.indices == inds

    # broadcast
    mask2 = Metale.SpPattern(size(spat))
    mask2 .= rand(Bool, size(spat))
    @test @inferred(spat .| mask2)::Metale.SpPattern == Array(spat) .| Array(mask2)
    @test @inferred(spat .& mask2)::Metale.SpPattern == Array(spat) .& Array(mask2)
    @test @inferred(spat .| Array(mask2))::Metale.SpPattern == Array(spat) .| Array(mask2)
    @test @inferred(spat .& Array(mask2))::Metale.SpPattern == Array(spat) .& Array(mask2)

    # fill!
    fill!(spat, false)
    @test all(==(false), spat)
end

@testset "SpArray" begin
    A = (@inferred Metale.SpArray{Float64}((5,5)))::Metale.SpArray{Float64, 2, Vector{Float64}}
    A = (@inferred Metale.SpArray{Float64}(5,5))::Metale.SpArray{Float64, 2, Vector{Float64}}

    @test all(==(0), A)
    for i in eachindex(A)
        # @test_throws Exception A[i] = 1
    end

    B = Metale.SpArray{Int}(5,5)
    A_spat = rand(Bool, size(A))
    B_spat = rand(Bool, size(B))

    for (x, x_spat) in ((A, A_spat), (B, B_spat))
        x.spat .= x_spat
        Metale.reinit!(x)
        @test x.spat == x_spat
        @test count(x.spat) == length(x.data)
        for i in eachindex(x)
            if x_spat[i]
                x[i] = i
                @test x[i] == i
            else
                # @test_throws Exception x[i] = i
            end
        end
    end

    # broadcast
    AA = Array(A)
    BB = Array(B)
    @test @inferred(A + A)::Metale.SpArray{Float64} == AA + AA
    @test @inferred(A + B)::Metale.SpArray{Float64} == map((x,y) -> ifelse(x==0||y==0,0,x+y), AA, BB)
    @test @inferred(A .* A)::Metale.SpArray{Float64} == AA .* AA
    @test @inferred(A .* B)::Metale.SpArray{Float64} == map((x,y) -> ifelse(x==0||y==0,0,x*y), AA, BB)
    @test @inferred(broadcast!(*, A, A, A))::Metale.SpArray{Float64} == broadcast!(*, AA, AA, AA)
    @test @inferred(broadcast!(*, A, A, B))::Metale.SpArray{Float64} == broadcast!(*, AA, AA, BB)
    @test A.spat == A_spat # sparsity pattern is never changed in `broadcast`
    @test @inferred(broadcast!(*, A, AA, B, 2))::Metale.SpArray{Float64} == broadcast!(*, AA, AA, BB, 2)
    @test A.spat == A_spat # sparsity pattern is never changed in `broadcast`
end
