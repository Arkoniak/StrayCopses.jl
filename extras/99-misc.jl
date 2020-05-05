using Revise
using StrayCopses
using CSV
using StatsBase

using StrayCopses: Node, DecisionTreeContainer, process_node

function prepare_dataset(df, k = 50, n = 3)
    if n == 3
        ids = (1:k, 51:(50+k), 101:(100+k))
    elseif n == 2
        ids = (1:k, 51:(50+k))
    else
        ids = 1:k
    end
    ids = collect(Iterators.flatten(ids))
    X = convert(Matrix, df[ids, 1:size(df, 2) - 1])
    y = convert(Vector, df[ids, size(df, 2)])

    out_classes = unique(y)
    n_classes = length(out_classes)
    in_classes = collect(1:n_classes)
    d = Dict(zip(out_classes, in_classes))
    target = map(z -> d[z], y)
    T = eltype(X)

    return (X = X, y = y, out_classes = out_classes, n_classes = n_classes, target = target, T = T)
end

df = CSV.read(joinpath(@__DIR__, "data", "uci_iris_all.csv"))
X, y, out_classes, n_classes, target, T = prepare_dataset(df);

function build_tree(X, y; max_depth = 1000000, n_features = 4, n_classes = 3)
    T = eltype(X)
    root = Node{T}()
    dtc = DecisionTreeContainer{T}(root, n_features, n_classes, max_depth, 1)
    process_node(dtc, root, X, y)

    return root
end

function lookahead(tree::Node, X)
    res = similar(target)
    for i in axes(target, 1)
        res[i] = predict(tree, X, i)
    end
    res
end

function accuracy(x1, x2)
    sum(x1 .== x2)/length(x1)
end

begin
    tree = build_tree(X, target, max_depth = 1)
    yhat = lookahead(tree, X)
    accuracy(target, yhat)
end

begin
    tree = build_tree(X, target, max_depth = 1_000_000)
    yhat = lookahead(tree, X)
    accuracy(target, yhat)
end

begin
    sc = StrayCopse()
    yhat = predict(md, X)
    md = fit(sc, X, y; max_depth = 3, n_trees = 10)
    accuracy(yhat, y)
end

tree

X0 = [6.5 2.8 4.6 1.5; 6.5 3.0 5.8 2.2]
y0 = [2, 3]

StrayCopses.get_split_indices(X0, 1, 6.5)

StrayCopses.best_split(X0, y0, dtc.n_classes, 1:4)

StrayCopses.feature_best_split(X0, y0, dtc.n_classes, 2)

axes(target)
