using Pkg

Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

include("runtests.jl")
