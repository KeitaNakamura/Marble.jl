struct NodeState
    a::Float64
    b::Float64
end

@testset "Grid" begin
    # constructors
    @test @inferred(Grid(0:10))::Grid{1, Float64} == Vec.(0:10)
    @test @inferred(Grid(0:10, 0:20))::Grid{2, Float64} == Vec.(collect(Iterators.product(0:10, 0:20)))
    @test @inferred(Grid(0:10, 0:20, 0:30))::Grid{3, Float64} == Vec.(collect(Iterators.product(0:10, 0:20, 0:30)))
    @test @inferred(Grid(LinearBSpline(), 0:10))::Grid{1, Float64} == Grid(0:10)
    @test @inferred(Grid(LinearBSpline(), 0:10, 0:20))::Grid{2, Float64} == Grid(0:10, 0:20)
    @test @inferred(Grid(LinearBSpline(), 0:10, 0:20, 0:30))::Grid{3, Float64} == Grid(0:10, 0:20, 0:30)
    @test_throws MethodError Grid(NodeState, 0:10)
    @test_throws MethodError Grid(NodeState, 0:10, 0:20)
    @test_throws MethodError Grid(NodeState, 0:10, 0:20, 0:30)
    @test @inferred(Grid(NodeState, LinearWLS(LinearBSpline()), 0:10))::Grid{1, Float64} == Grid(0:10)
    @test @inferred(Grid(NodeState, LinearWLS(LinearBSpline()), 0:10, 0:20))::Grid{2, Float64} == Grid(0:10, 0:20)
    @test @inferred(Grid(NodeState, LinearWLS(LinearBSpline()), 0:10, 0:20, 0:30))::Grid{3, Float64} == Grid(0:10, 0:20, 0:30)

    # gridsteps/gridaxes/gridorigin
    grid = Grid(CubicBSpline(), 0:1.0:10, 1:2.0:20)
    @test @inferred(gridsteps(grid)) == (1.0, 2.0)
    @test @inferred(gridsteps(grid, 1)) == 1.0
    @test @inferred(gridsteps(grid, 2)) == 2.0
    @test @inferred(gridaxes(grid)) == (0:1.0:10, 1:2.0:20)
    @test @inferred(gridaxes(grid, 1)) == 0:1.0:10
    @test @inferred(gridaxes(grid, 2)) == 1:2.0:20
    @test @inferred(gridorigin(grid))::Vec{2} == Vec(0,1)

    # neighboring_nodes/neighboring_cells/whichcell/whichblock
    @test @inferred(Metale.neighboring_nodes(grid, Vec(0.6, 8.8), 1))::CartesianIndices == CartesianIndices((1:2, 4:5))
    @test @inferred(Metale.neighboring_nodes(grid, Vec(0.6, 8.8), 2))::CartesianIndices == CartesianIndices((1:3, 3:6))
    @test @inferred(Metale.neighboring_nodes(grid, Vec(0.6, 8.8), Vec(1,2)))::CartesianIndices == CartesianIndices((1:2, 3:6))
    @test @inferred(Metale.neighboring_nodes(grid, Vec(0.6, 8.8)))::CartesianIndices == CartesianIndices((1:3, 3:6))
    @test @inferred(Metale.neighboring_nodes(grid, Vec(-0.6, 8.8)))::CartesianIndices == CartesianIndices((1:0, 1:0))
    @test @inferred(Metale.neighboring_cells(grid, Vec(0.6, 8.8), 1))::CartesianIndices == CartesianIndices((1:2, 3:5))
    @test @inferred(Metale.neighboring_cells(grid, Vec(0.6, 8.8), 2))::CartesianIndices == CartesianIndices((1:3, 2:6))
    @test_throws BoundsError Metale.neighboring_cells(grid, CartesianIndex(11, 9), 1)
    @test @inferred(Metale.neighboring_blocks(grid, Vec(8.8, 4.6), 1))::CartesianIndices == CartesianIndices((1:2, 1:2))
    @test @inferred(Metale.neighboring_blocks(grid, Vec(8.8, 4.6), 2))::CartesianIndices == CartesianIndices((1:2, 1:2))
    @test_throws BoundsError Metale.neighboring_blocks(grid, CartesianIndex(3, 1), 1)
    @test (Metale.whichcell(grid, Vec(0.6, 8.8)))::CartesianIndex == CartesianIndex(1, 4)
    @test (Metale.whichcell(grid, Vec(-0.6, 8.8)))::Nothing == nothing
    @test (Metale.whichblock(grid, Vec(8.8, 4.6)))::CartesianIndex == CartesianIndex(2, 1)
    @test (Metale.whichblock(grid, Vec(-8.8, 4.6)))::Nothing == nothing

    # pointsinblock
    @test Metale.blocksize(grid) == (2, 2)
    xₚ = Vec{2, Float64}[(2,2), (8, 18), (8, 21), (4, 18), (5, 18)]
    @test Metale.pointsinblock(grid, xₚ) == reshape([[1], [], [4, 5], [2]], 2, 2)

    # threadsafe_blocks
    @test Metale.threadsafe_blocks((20, 30)) == [[CartesianIndex(1,1) CartesianIndex(1,3); CartesianIndex(3,1) CartesianIndex(3,3)],
                                                 [CartesianIndex(2,1) CartesianIndex(2,3)],
                                                 [CartesianIndex(1,2) CartesianIndex(1,4); CartesianIndex(3,2) CartesianIndex(3,4)],
                                                 [CartesianIndex(2,2) CartesianIndex(2,4)]]
end
