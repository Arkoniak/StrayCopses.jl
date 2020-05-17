module TestDecisionTree
using Test
using StrayCopses
using StrayCopses: feature_best_split, best_split, Node, DecisionTreeContainer, process_node, create_containers
using StableRNGs

@testset "Basic split" begin
    X = reshape([1., 2., 3., 4.], :, 1)
    y = [1, 1, 2, 2]

    containers = create_containers(2, y)
    @test feature_best_split(containers, X, y, 2, 1) == (val = 3.000, impurity = 0.5)

    rng = StableRNG(2020)
    N = 5
    n_classes = 3
    X = reshape(rand(rng, 2*N), :, 2)
    y = rand(rng, 1:n_classes, N)
    containers = StrayCopses.create_containers(n_classes, y)
    
    res = feature_best_split(containers, X, y, n_classes, 1)
    @test res.val ≈ 0.257183060127947
    @test res.impurity ≈ 0.2599999999999999

    res = feature_best_split(containers, X, y, n_classes, 2)
    @test res.val ≈ 0.9442017794271014
    @test res.impurity ≈ 0.36
end

@testset "Best split" begin
    X = collect(reshape([1., 2., 3., 4., 1., 2., 2., 4.], :, 2))
    y = [1, 1, 2, 2]

    containers = create_containers(2, y)
    @test best_split(X, y, 2, 1:2) == (feature = 1, val = 3.0000)
    @test best_split(X, y, 2, 1:1) == (feature = 1, val = 3.0000)
    @test best_split(X, y, 2, [2]) == (feature = 2, val = 2.0000)

    X = [6.5 2.8 4.6 1.5; 6.5 3.0 5.8 2.2]
    y = [2, 3]
    @test best_split(X, y, 3, 1:4) == (feature = 2, val = 3.0)
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
    @test tree == "[X4 < 66.0]\n↦[X10 < 17.0]\n⎸↦[X3 < 78.0]\n⎸⎸↳[1]\n⎸⎸↳[2]\n⎸↳[1]\n↦[X4 < 83.0]\n ↳[2]\n ↳[3]\n"
end

end # module
