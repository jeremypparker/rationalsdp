using Pkg

Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

include("testutils.jl")
include("slowtest_helpers.jl")
include("slowtests.jl")
