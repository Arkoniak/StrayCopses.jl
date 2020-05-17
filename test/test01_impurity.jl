module TestImpurity
using Test
using StrayCopses
using StrayCopses: information_gain, information_impurity, gini_index, gini_impurity

@testset "Basic information gain" begin
    @test information_gain([3, 3], 6) == 1.0
    @test information_gain([2, 6], 8) ≈ 0.8112781244591328
    information_before = information_gain([5, 9], 14)
    @test information_impurity(information_before, [3, 3], [2, 6], 6, 8, 14) ≈ 0.04812703040826927
end

@testset "Basic gini index" begin
    @test gini_index([4, 2], 6) ≈ 0.555555555555555555555
    @test gini_index([2, 2], 4) ≈ 0.5
    @test gini_index([2, 0], 2) ≈ 1
    gini_before = gini_index([4, 2], 6)
    @test gini_impurity(gini_before, [2, 2], [2, 0], 4, 2, 6) ≈ 0.111111111111111111111
end

@testset "Edge case gini index" begin
    gini_before = gini_index([2, 2], 4)
    @test gini_impurity(gini_before, [0, 0], [2, 2], 0, 4, 4) ≈ 0.
    @test gini_index([0, 0], 0) ≈ 0.
end

end # module
