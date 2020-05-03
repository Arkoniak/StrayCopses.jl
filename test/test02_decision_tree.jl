module TestDecisionTree
using Test
using StrayCopse
using StrayCopse: feature_best_split, best_split

@testset "Basic split" begin
    X = reshape([1., 2., 3., 4.], :, 1)
    y = [1, 1, 2, 2]

    @test feature_best_split(X, y, 2, 1) ≈ 3.000
end

@testset "Best split" begin
    X = collect(reshape([1., 2., 3., 4., 1., 2., 2., 4.], :, 2))
    y = [1, 1, 2, 2]

    @test best_split(X, y, 2, 1:2) == (1, 3.0000)
    @test best_split(X, y, 2, 1:1) == (1, 3.0000)
    @test best_split(X, y, 2, [2]) == (2, 2.0000)
end

end # module
