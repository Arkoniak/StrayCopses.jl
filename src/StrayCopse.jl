module StrayCopse

using StatsBase

include("impurity.jl")
include("decision_tree.jl")
include("copse.jl")

export predict, fit, StrayCopseClassifier

end # module
