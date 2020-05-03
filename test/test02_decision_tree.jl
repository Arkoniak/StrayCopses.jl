module TestDecisionTree
using Test
using StrayCopses
using StrayCopses: feature_best_split, best_split, Node, DecisionTreeContainer, process_node
using StableRNGs

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

@testset "random dataset" begin
    rng = StableRNG(2020)
    X = rand(rng, 10, 10)
    X = floor.(X .* 100)
    y = rand(rng, 1:3, 10)
    T = eltype(X)

    root = Node{T}()
    dtc = DecisionTreeContainer{T}(root, 10, 3, 6, 1)
    process_node(dtc, root, X, y, rng, 1:10)
    io = IOBuffer()
    print(io, root)
    tree = String(take!(io))
    @test tree == "[X7 < 95.0]\n↦[X8 < 91.0]\n⎸↦[X5 < 95.0]\n⎸⎸↦[X4 < 88.0]\n⎸⎸⎸↳[1]\n⎸⎸⎸↳[3]\n⎸⎸↦[X5 < 99.0]\n⎸⎸ ↳[2]\n⎸⎸ ↳[1]\n⎸↳[3]\n↳[2]\n"
end

end # module
