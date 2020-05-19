mutable struct Node{T}
    feature_idx::Int
    feature_val_idx::UInt32
    feature_val::T
    value::Int
    left::Node
    right::Node
    is_terminal::Bool

    function Node(feature_idx, feature_val_idx, feature_val::T) where {T}
        node = new{T}()
        node.feature_idx = feature_idx
        node.feature_val_idx = feature_val_idx
        node.feature_val = feature_val
        node.is_terminal = false

        return node
    end

    # function Node{T}(value) where {T}
    #     node = new{T}()
    #     node.value = value
    #     node.is_terminal = true

    #     return node
    # end

    # function Node(feature_idx, feature_val::T, value, is_terminal=false) where {T}
    #     node = new{T}()
    #     node.feature_idx = feature_idx
    #     node.value = value
    #     node.feature_val = feature_val
    #     node.is_terminal = is_terminal

    #     return node
    # end

    # function Node{T}() where T
    #     node = new{T}()
    #     node.is_terminal = false
    #     return node
    # end
end

struct DecisionTree
    n_features_per_node::Int
    n_classes::Int
    max_depth::Int
    min_node_records::Int
end

"""
    feature_best_split

For a given feature search best split value.
"""
function feature_best_split(containers, X, y, n_classes, feature)
    gini_before = containers.gini_before
    left = containers.left
    right = containers.right
    lt = containers.lt
    ids = containers.ids
    perms = X.perms
    ranks = X.ranks

    # prepare initial split
    ll = 1
    lr = length(y) - 1
    i1 = X.perms[ids[1], feature]
    left[y[i1]] = 1
    right[y[i1]] -= 1
    prev_val = ranks[i1, feature]
    best_val = prev_val
    best_impurity = 0.0
    @inbounds for idx in 2:length(ids)
        i = perms[ids[idx], feature]
        if ranks[i, feature] != prev_val
            prev_val = ranks[i, feature]
            impurity = gini_impurity(gini_before, left, right, ll, lr, lt)
            if impurity > best_impurity
                best_impurity = impurity
                best_val = prev_val
            end
        end 
        ll += 1
        lr -= 1
        left[y[i]] += 1
        right[y[i]] -= 1
    end

    return (val = best_val, impurity = best_impurity)
end

function create_containers(n_classes, y, ids)
    left = zeros(Int, n_classes)
    right = zeros(Int, n_classes)
    lt = length(ids)
    for i in ids
        right[y[i]] += 1
    end
    gini_before = gini_index(right, lt)
    containers = (left = left, right = right, gini_before = gini_before, lt = lt, ids = ids)
 
    return containers
end

# Chooses best feature from features
function best_split(X, y, ids, n_classes, features)
    containers = create_containers(n_classes, y, ids)
    best_feature = 0
    best_val = -Inf
    best_impurity = -Inf
    for feature in features
        val, impurity = feature_best_split(containers, X, y, n_classes, feature)
        if impurity > best_impurity
            best_val = val
            best_feature = feature
            best_impurity = impurity
        end
    end

    return (feature = best_feature, val = best_val)
end

function split_value(X, target, n_classes)
    res = zeros(Int, n_classes)
    for i in axes(X, 1)
        res[target[i]] += 1
    end

    return argmax(res)
end

function get_split_indices(X, feature_idx, feature_val)
    return X[:, feature_idx] .< feature_val, X[:, feature_idx] .>= feature_val
end

function is_pure(target)
    return all(target[1] .== target)
end

###############################
# Node functions
###############################

function build_node(X, y, n_classes, n_features_per_node,
                   rng = Random.GLOBAL_RNG,
                    ids = axes(X, 1),
                   features = sample(rng, 1:size(X, 2), n_features_per_node, replace = false))
    # Ok, this one is type stable, since we always return indices of corresponding values
    feature_idx, feature_val_idx = best_split(X, y, ids, n_classes, features)
    feature_val = X.X[feature_val_idx, feature_idx]

    root = Node(feature_idx, feature_val_idx, feature_val)
    return root

function process_node(dt::DecisionTree, node, X, target, ids,
                      rng = Random.GLOBAL_RNG, 
                      depth = 1)

    if depth > dt.max_depth
        node.is_terminal = true
        node.value = split_value(X, target, ids, dt.n_classes)
    elseif length(ids) <= dtc.min_node_records
        node.is_terminal = true
        node.value = split_value(X, target, ids, dt.n_classes)
    elseif is_pure(target, ids)
        node.is_terminal = true
        node.value = target[ids[1]]
    else
        # feature_idx, feature_val = best_split(X, target, dt.n_classes, features)
        # node.feature_idx = feature_idx
        # node.feature_val = feature_val
        left_ids, right_ids = get_split_indices(X, node.feature_idx, node.feature_val_idx, ids)

        # process left node
        node.left = build_node(X, y, n_classes, dt.n_features_per_node, rng, left_ids)
        node.right = build_node(X, y, n_classes, dt.n_features_per_node, rng, right_ids)
        process_node(dt, node.left, X, y, left_ids, rng, depth + 1)
        process_node(dt, node.right, X, y, right_ids, rng, depth + 1)
    end
    # new_features = sample(rng, 1:size(X, 2), dt.n_features_per_node, replace = false)
    # feature_idx, feature_val_idx = best_split(X, y, n_classes, new_features, left_ids)
    # feature_val = X.X[feature_val_idx, feature_idx]
    # left = Node(feature_idx, feature_val_idx, feature_val)

    # process right node
    # new_features = sample(rng, 1:size(X, 2), dt.n_features_per_node, replace = false)
    # feature_idx, feature_val_idx = best_split(X, y, n_classes, new_features, right_ids)
    # feature_val = X.X[feature_val_idx, feature_idx]
    # right = Node(feature_idx, feature_val_idx, feature_val)

    # node.left = left
    # node.right = right

    # left = process_node(dt, )
    # left = Node{T}()
    # right = Node{T}()
    # node.left = left
    # node.right = right
    # new_features = sample(rng, 1:size(X, 2), dt.n_features_per_node, replace = false)
    # process_node(dt, left, X[left_ids, :], target[left_ids], rng, new_features, depth + 1)
    # process_node(dt, right, X[right_ids, :], target[right_ids], rng, new_features, depth + 1)

    # if depth > dtc.max_depth
    #     node.is_terminal = true
    #     node.value = split_value(X, target, dtc.n_classes)
    # elseif length(target) <= dtc.min_node_records
    #     node.is_terminal = true
    #     node.value = split_value(X, target, dtc.n_classes)
    # elseif is_pure(target)
    #     node.is_terminal = true
    #     node.value = target[1]
    # else
        # feature_idx, feature_val = best_split(X, target, dtc.n_classes, features)
        # node.feature_idx = feature_idx
        # node.feature_val = feature_val
        # left_ids, right_ids = get_split_indices(X, feature_idx, feature_val)
        # left = Node{T}()
        # right = Node{T}()
        # node.left = left
        # node.right = right
        # new_features = sample(rng, 1:size(X, 2), dtc.n_features_per_node, replace = false)
        # process_node(dtc, left, X[left_ids, :], target[left_ids], rng, new_features, depth + 1)
        # process_node(dtc, right, X[right_ids, :], target[right_ids], rng, new_features, depth + 1)
end

function create_tree(X, y; rng = Random.GLOBAL_RNG, max_depth = typemax(Int),
                     min_node_records = 1, n_features = size(X, 2))
    T = eltype(X)
    n_classes = length(Set(y))
    prm = PRMatrix(X)
    root = init_tree(prm, y, n_classes, n_features, rng)
    dt = DecisionTree(n_features, n_classes, max_depth, min_node_records)

    process_node(root, dt, prm, y, rng)
    
    return root
end

function predict(node::Node, row)
    if node.is_terminal
        return node.value
    else
        if row[node.feature_idx] < node.feature_val
            return predict(node.left, row)
        else
            return predict(node.right, row)
        end
    end
end

function predict(node::Node, X, i)
    if node.is_terminal
        return node.value
    else
        if X[i, node.feature_idx] < node.feature_val
            return predict(node.left, X, i)
        else
            return predict(node.right, X, i)
        end
    end
end

function Base.:show(io::IO, node::Node, prefix = "")
    if node.is_terminal
        if length(prefix) > 0
            new_prefix = collect(prefix)
            new_prefix[end] = '↳'
            new_prefix = join(new_prefix)
        else
            new_prefix = prefix
        end
        write(io, new_prefix, "[$(node.value)]", "\n")
    elseif !isdefined(node, :left)
        write(io, "[]\n")
    else
        if length(prefix) > 0
            new_prefix = collect(prefix)
            new_prefix[end] = '↦'
            new_prefix = join(new_prefix)
        else
            new_prefix = prefix
        end
        write(io, new_prefix, "[X$(node.feature_idx) < $(node.feature_val)]", "\n")
        left_prefix = prefix * "⎸"
        right_prefix = prefix * " "
        show(io, node.left, left_prefix)
        show(io, node.right, right_prefix)
    end
end
