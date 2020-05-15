using StrayCopses
using CSV
using BenchmarkTools

# Glass
d_train = CSV.read(joinpath(@__DIR__, "data", "uci_glass_train.csv"))
d_test = CSV.read(joinpath(@__DIR__, "data", "uci_glass_test.csv"))

X_train = convert(Matrix, d_train[:, 1:size(d_train, 2) - 1])
y_train = convert(Vector, d_train[!, size(d_train, 2)])
X_test = convert(Matrix, d_test[:, 1:size(d_test, 2) - 1])
y_test = convert(Vector, d_test[!, size(d_test, 2)])

sc = StrayCopse()
@btime md = fit($sc, $X_train, $y_train, max_depth = 1_000_000)
# 202.178 ms (1846793 allocations: 210.01 MiB)
md = fit(sc, X_train, y_train, max_depth = 1_000_000)
# 174.363 ms (1637322 allocations: 187.40 MiB)
@btime phat = predict($md, $X_test)
# 111.559 μs (2 allocations: 336 bytes)
phat = predict(md, X_test)

begin
    md = fit(sc, X_train, y_train, max_depth = 1_000_000)
    phat = predict(md, X_test)
    sum(y_test .== phat)/length(phat)
end

# Sonar
d_train = CSV.read(joinpath(@__DIR__, "data", "uci_sonar_train.csv"))
d_test = CSV.read(joinpath(@__DIR__, "data", "uci_sonar_test.csv"))

X_train = convert(Matrix, d_train[:, 1:size(d_train, 2) - 1])
y_train = convert(Vector, d_train[!, size(d_train, 2)])
X_test = convert(Matrix, d_test[:, 1:size(d_test, 2) - 1])
y_test = convert(Vector, d_test[!, size(d_test, 2)])

sc = StrayCopse()
@btime md = fit($sc, $X_train, $y_train)
# 419.147 ms (3499117 allocations: 375.54 MiB)
@btime phat = predict($md, $X_test)
# 111.559 μs (2 allocations: 336 bytes)

sum(y_test .== phat)/length(phat)

size(X_train)
md = fit(sc, X_train, y_train; max_depth = 1000000, n_trees = 500, n_features_per_node = 5)
phat = predict(md, X_test)

sum(y_test .== phat)/length(phat)

phat = predict(md, X_train)
sum(y_train .== phat)/length(phat)

r = 1:100
md = fit(sc, X_train[r, :], y_train[r]; max_depth = 10000, n_trees = 1, n_features_per_node = 60)
phat = predict(md, X_train[r, :])
sum(y_train[r] .== phat)/length(phat)
