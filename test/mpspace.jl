@testset "sparsity pattern" begin
    interp = LinearWLS(CubicBSpline())
    grid = Grid(0.0:1.0:10.0, 0.0:1.0:10.0)
    pointstate = generate_pointstate((x,y) -> y < 5.0, interp, grid)
    space = MPSpace(interp, grid, pointstate)
    mask = falses(size(grid))
    mask[:,6:end] .= true
    update!(space, pointstate; exclude = mask)
    @test !any(space.spat[:,7:end])
end
