module StrayCopses

using StatsBase
using Random

include("prmatrix.jl")
include("impurity.jl")
include("decision_tree.jl")
include("copse.jl")

export predict, fit, StrayCopse, create_tree, PRMatrix

end # module
