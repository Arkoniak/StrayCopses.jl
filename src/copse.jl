struct StrayCopse end

struct StrayCopseResult{T, S}
    sc::Vector{Node{T}}
    d::Vector{S}
end

function fit(::StrayCopse, X, y, rng = Random.GLOBAL_RNG; max_depth = 6, min_node_records = 1,
        n_features_per_node = Int(floor(sqrt(size(X, 2)))), n_trees = 100)

    if n_features_per_node > size(X, 2)
        @warn "n_features_per_node is $n_features_per_node which is larger than the overall features number $(size(X, 2))."
        n_features_per_node = size(X, 2)
    end

    out_classes = unique(y)
    n_classes = length(out_classes)
    in_classes = collect(1:n_classes)
    d = Dict(zip(out_classes, in_classes))
    target = map(z -> d[z], y)

    T = eltype(X)
    sc = Node{T}[]
    nrow = size(X, 1)
    for i in 1:n_trees
        root = Node{T}()
        push!(sc, root)
        ids = sample(1:nrow, nrow)
        X1 = X[ids, :]
        target1 = target[ids]
        dtc = DecisionTreeContainer{T}(root, n_features_per_node, n_classes, max_depth, min_node_records)
        process_node(dtc, root, X1, target1, rng)
    end

    return StrayCopseResult(sc, out_classes)
end

function predict(scr::StrayCopseResult, X)
    d = scr.d

    n = length(d)
    vals = Vector{Int}(undef, n)
    res = Vector{eltype(d)}(undef, size(X, 1))
    for i in axes(X, 1)
        vals .= 0
        for tree in scr.sc
            vals[predict(tree, X, i)] += 1
        end
        res[i] = d[argmax(vals)]
    end

    return res
end
