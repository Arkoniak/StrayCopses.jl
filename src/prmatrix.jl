# Despite I call this structure `Matrix`, it has nothing in common with matrices.
# At least I can't think of any meaningful usage, yet it's name contains matrix, since
# it is wrapper around original input matrix 
struct PRMatrix{T}
    X::T
    
    # each element of permutation matrix relates to the index of the element
    # of the input matrix `X`. I hope that this implementation will never be used
    # for sample which have more than 4 billions elements. Anyway if someone 
    # tries to solve a problem with the input matrix of this size, he has bigger
    # problems than changing UInt32 to UInt64
    perms::Matrix{UInt32}
    ranks::Matrix{UInt32}

    # TODO: currently there is only two modes (either ordinal or categorical vairables),
    # so I could have used BitArray, but that limit me to these two modes, and require 
    # potential substantial rewriting of the code, so instead of BitArray, I use 
    # 1 byte array (with potentially 254 additional modes). At the same time, I greatly
    # dislike the idea of conditional hardcoded mode, so in the future it should be 
    # changed to a more flexible array of objects of particular type. Potentially it
    # can lead to flexible "strategies", when you can choose way how feature vector is
    # being split just by supplying new type and dispatching corresponding 
    # best_feature_split method.

    # Currently 0 means "le" strategy (X < split_val), which is suitable for ordinal 
    # variables and 1 means "eq" strategy (X == split_val) which is suitable for 
    # categorical variables.
    modes::Vector{UInt8}
end

PRMatrix(X::PRMatrix) = X
function PRMatrix(X)
    nrow, ncol = size(X)
    perms = Matrix{UInt32}(undef, nrow, ncol)
    ranks = Matrix{UInt32}(undef, nrow, ncol)
    for i in 1:ncol
        sortperm!(@view(perms[:, i]), @view(X[:, i]))
        ranks[perms[1, i], i] = 1
        rank = 1
        prev_val = X[perms[1, i], i]
        for j in 2:nrow
            idx = perms[j, i]
            if X[idx, i] != prev_val
                prev_val = X[idx, i]
                rank += 1
            end
            ranks[idx, i] = rank
        end
    end
    modes = zeros(UInt8, ncol)

    return PRMatrix{T}(X, perms, ranks, modes)
end
