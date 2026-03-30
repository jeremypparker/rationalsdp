module RationalSDP

using DoubleFloats: Double64
using LinearAlgebra
using Printf
using Base.Threads
import MathOptInterface as MOI

include("optimizer.jl")

export Optimizer, Settings

end
