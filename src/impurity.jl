# here we assume, that groups is a list of cnt of each value
# of the target variable
function node_group_information_gain(group)
    den = sum(group)
    res = 0.0
    for x in group
        res -= x/den * log2(x/den)
    end
    return res
end

function node_information_gain(groups)
    den = sum(sum.(groups))
    res = 0.0
    for group in groups
        res += sum(group)/den * node_group_information_gain(group)
    end
    return res
end

function information_gain(groups)
    classes = sum(groups)
    whole_node = node_group_information_gain(classes)
    return whole_node - node_information_gain(groups)
end

function node_group_gini_index(group)
    den = sum(group)
    den == 0 && return 0.0
    res = 0.0
    for x in group
        res += (x/den)^2
    end
    return 1.0 - res
end

function node_gini_index(groups)
    den = sum(sum.(groups))
    res = 0.0
    for group in groups
        res += sum(group)/den * node_group_gini_index(group)
    end
    return res
end

function node_group_gini_index(group, den)
    den == 0 && return 0.0
    res = 0.0
    for x in group
        res += (x/den)^2
    end
    return res
end

function node_gini_index(left, right, ll, lr, lt)
    res  = ll/lt * node_group_gini_index(left, ll)
    res += lr/lt * node_group_gini_index(right, lr)

    return res
end

function gini_index(groups)
    classes = sum(groups)
    whole_node = node_group_gini_index(classes)
    return whole_node - node_gini_index(groups)
end

function gini_impurity(gini_before, left, right, ll, lr, lt)
    return node_gini_index(left, right, ll, lr, lt) - gini_before
end
