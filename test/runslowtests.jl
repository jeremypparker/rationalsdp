using Pkg

Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

include("testutils.jl")
include("kse_timeaverage_helpers.jl")
include("slowtests.jl")
