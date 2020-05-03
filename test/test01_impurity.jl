module TestImpurity
using Test
using StrayCopses
using StrayCopses: node_group_information_gain, node_information_gain, information_gain
using StrayCopses: node_group_gini_index, node_gini_index, gini_index

@testset "Basic information gain" begin
    @test node_group_information_gain([3, 3]) == 1.0
    @test node_group_information_gain([2, 6]) ≈ 0.8112781244591328
    @test node_information_gain([[3, 3], [2, 6]]) ≈ 0.8921589282623617
    @test information_gain([[3, 3], [2, 6]]) ≈ 0.04812703040826927
end

@testset "Basic gini index" begin
    @test node_group_gini_index([4, 2]) ≈ 0.444444444444444444
    @test node_group_gini_index([2, 2]) ≈ 0.5
    @test node_group_gini_index([2, 0]) ≈ 0
    @test node_gini_index([[2, 2], [2, 0]]) ≈ 0.333333333333333333
    @test gini_index([[2, 2], [2, 0]]) ≈ 0.111111111111111111111
end

@testset "Edge case gini index" begin
    @test gini_index([[0, 0], [2, 2]]) ≈ 0.
    @test node_group_gini_index([0, 0]) ≈ 0.
end

end # module
