# Split the solver into focused files so each subsystem is easier to navigate.

include("optimizer/core.jl")
include("optimizer/algebra.jl")
include("optimizer/barrier.jl")
include("optimizer/extraction.jl")
include("optimizer/phases.jl")
include("optimizer/solve.jl")
