"""
    test_split

For a given feature and value count number of examples that goes to the left and 
right branch correspondingly
"""
function test_split(containers, X, target, n_classes, feature, value)
    left = containers.left
    right = containers.right
    left .= 0
    right .= 0
    @inbounds @simd for i in axes(X, 1)
        if X[i, feature] < value
            left[target[i]] += 1
        else
            right[target[i]] += 1
        end
    end
end

"""
    feature_best_split

For a given feature search best split value.
"""
function feature_best_split(containers, X, target, n_classes, feature)
    gini_before = containers.gini_before
    left = containers.left
    right = containers.right
    lt = containers.lt

    best_val = -Inf
    best_impurity = -Inf
    @inbounds for i in axes(X, 1)
        test_split(containers, X, target, n_classes, feature, X[i, feature])
        ll = sum(left)
        lr = sum(right)
        impurity = gini_impurity(gini_before, left, right, ll, lr, lt)
        if impurity > best_impurity
            best_impurity = impurity
            best_val = X[i, feature]
        end
    end

    return (val = best_val, impurity = best_impurity)
end
