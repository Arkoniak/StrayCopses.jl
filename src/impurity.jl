# here we assume, that groups is a list of cnt of each value
# of the target variable
function information_gain(group, den)
    res = 0.0
    for x in group
        p = x/den
        res -= p * log2(p)
    end
    return res
end

function information_impurity(information_before, left, right, ll, lr, lt)
    information_before - 
    ll/lt * information_gain(left, ll) -
    lr/lt * information_gain(right, lr)
end

function gini_index(group, den)
    den == 0 && return 0.0
    res = 0.0
    for x in group
        res += (x/den)^2
    end
    return res
end

function gini_impurity(gini_before, left, right, ll, lr, lt)
    ll/lt * gini_index(left, ll) +
    lr/lt * gini_index(right, lr) -
    gini_before
end
