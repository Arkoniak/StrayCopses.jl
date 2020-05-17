module TestPRMatrix
using Test
using StrayCopses

@testset "micro input matrix" begin
    X = reshape([0.2 0.1], :, 1)
    prm = PRMatrix(X)

    @test all(prm.perms .== reshape([2, 1], :, 1))
    @test all(prm.ranks .== reshape([2, 1], :, 1))

    X = reshape([0.9, 0.1, 0.1, 0.9], :, 1)
    prm = PRMatrix(X)
    @test all(prm.perms .== reshape([2, 3, 1, 4], :, 1))
    @test all(prm.ranks .== reshape([2, 1, 1, 2], :, 1))
end

@testset "small input matrix" begin
    X = [0.9 0.1; 0.5 0.2; 0.5 0.4; 0.9 0.3]
    prm = PRMatrix(X)

    @test all(prm.perms .== [2 1; 3 2; 1 4; 4 3])
    @test all(prm.ranks .== [2 1; 1 2; 1 4; 2 3])
end

@testset "mixed input matrix" begin
    X = [0.9 0.1 "a"; 0.5 0.2 "b"; 0.5 0.4 "a"; 0.9 0.3 "b"]
    prm = PRMatrix(X)

    @test all(prm.perms .== [2 1 1; 3 2 3; 1 4 2; 4 3 4])
    @test all(prm.ranks .== [2 1 1; 1 2 2; 1 4 1; 2 3 2])
end

end # module
