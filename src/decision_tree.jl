mutable struct Node{T}
    feature_idx::Int
    feature_val::T
    value::Int
    left::Node{T}
    right::Node{T}
    is_terminal::Bool

    function Node(feature_idx, feature_val::T) where {T}
        node = new{T}()
        node.feature_idx = feature_idx
        node.feature_val = feature_val
        node.is_terminal = false

        return node
    end

    function Node{T}(value) where {T}
        node = new{T}()
        node.value = value
        node.is_terminal = true

        return node
    end

    function Node(feature_idx, feature_val::T, value, is_terminal=false) where {T}
        node = new{T}()
        node.feature_idx = feature_idx
        node.feature_val = feature_val
        node.value = value
        node.is_terminal = is_terminal

        return node
    end

    function Node{T}() where T
        node = new{T}()
        node.is_terminal = false
        return node
    end
end

struct DecisionTreeContainer{T}
    root::Node{T}
    n_features_per_node::Int
    n_classes::Int
    max_depth::Int
    min_node_records::Int
end

function test_split(X, target, n_classes, feature, value)
    left = zeros(Int, n_classes)
    right = zeros(Int, n_classes)
    for i in axes(X, 1)
        if X[i, feature] < value
            left[target[i]] += 1
        else
            right[target[i]] += 1
        end
    end

    return left, right
end

# find best split for given feature
function feature_best_split(X, target, n_classes, feature)
    best_val = -Inf
    best_impurity = -Inf
    for i in axes(X, 1)
        left, right = test_split(X, target, n_classes, feature, X[i, feature])
        impurity = gini_index([left, right])
        if impurity > best_impurity
            best_impurity = impurity
            best_val = X[i, feature]
        end
    end

    return (val = best_val, impurity = best_impurity)
end

# Chooses best feature from features
function best_split(X, target, n_classes, features)
    best_feature = 0
    best_val = -Inf
    best_impurity = -Inf
    for feature in features
        val, impurity = feature_best_split(X, target, n_classes, feature)
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

function process_node(dtc::DecisionTreeContainer{T}, node, X, target, rng = Random.GLOBAL_RNG,
        features = sample(rng, 1:size(X, 2), dtc.n_features_per_node, replace = false),
        depth = 1) where T

    if depth > dtc.max_depth
        node.is_terminal = true
        node.value = split_value(X, target, dtc.n_classes)
    elseif length(target) <= dtc.min_node_records
        node.is_terminal = true
        node.value = split_value(X, target, dtc.n_classes)
    elseif is_pure(target)
        node.is_terminal = true
        node.value = target[1]
    else
        feature_idx, feature_val = best_split(X, target, dtc.n_classes, features)
        node.feature_idx = feature_idx
        node.feature_val = feature_val
        left_ids, right_ids = get_split_indices(X, feature_idx, feature_val)
        # if (size(X) == (2, 4)) & (feature_idx == 1) & (feature_val == 6.5) & (length(left_ids) == 2) & (length(right_ids) == 2)
        #     @show X, target
        #     return
        # end
        # @show size(X), feature_idx, feature_val, length(left_ids), length(right_ids)
        left = Node{T}()
        right = Node{T}()
        node.left = left
        node.right = right
        new_features = sample(rng, 1:size(X, 2), dtc.n_features_per_node, replace = false)
        process_node(dtc, left, X[left_ids, :], target[left_ids], rng, new_features, depth + 1)
        process_node(dtc, right, X[right_ids, :], target[right_ids], rng, new_features, depth + 1)
    end
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
