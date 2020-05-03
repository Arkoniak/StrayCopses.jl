module TestStrayCopse
using Test
using StableRNGs
using StrayCopses

@testset "basic stray copse" begin
    X = collect(reshape([1., 2., 3., 4., 1., 2., 2., 4.], :, 2))
    y = [1, 1, 2, 2]
    rng = StableRNG(2020)

    res = fit(StrayCopse(), X, y, rng, n_features_per_node = 2, n_trees = 2)
    @test length(res.sc) == 2
end

end # module
