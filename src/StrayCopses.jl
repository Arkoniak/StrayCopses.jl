module StrayCopses

using StatsBase

include("impurity.jl")
include("decision_tree.jl")
include("copse.jl")

export predict, fit, StrayCopse

end # module
