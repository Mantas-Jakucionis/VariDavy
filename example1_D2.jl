using Plots; default(size=(1200,800), thickness_scaling=2)
using FFTW
using DelimitedFiles
using Distributed

### For use with HPC nodes
# using SlurmClusterManager
# addprocs(SlurmManager())

@everywhere begin
using Plots
using Dates
using LinearAlgebra
using MKL
using Random
using StatsBase
using Distributions
using Tullio
using Parameters
using OrdinaryDiffEq
using DiffEqCallbacks
using Combinatorics
using OffsetArrays
using IterativeSolvers
using LightGraphs
using BenchmarkTools

using Optimization
using OptimizationBBO

using OptimizationOptimJL
using QuadGK

using LaTeXStrings

BLAS.set_num_threads(1)

include("Structures.jl")
include("Auxilary.jl")
include("Propagators.jl")
include("Observables.jl")
include("mD2_Model.jl")
include("State.jl")
include("Inputs.jl")
include("FeynmanDiagrams.jl")
include("FeynmanDiagrams_Threaded.jl")
end

input = Example_Model(

                    nmultp = 1,
                    nsite = 1,
                    ntraj = 1,
                    tmax = 2000.0,
                    dt = 0.25,
                    
                    bathSpectralDensity = Drude,
                    bathSpdParameters = [100],
                    
                    wmin = 0.1,
                    wmax = 500.0,
                    dw = 5,
                    
                    temp = 0,

                    twaiting2D = 0.0,
                    tmax2D     = 300.0,
                    dt2D       = 10.0
                    
                    )

J = -100.0

s = input.nsite
s > 1 ? (@tullio  input.H[i,j] = δ(i,j-1) + δ(i,j+1) + δ(i,δ(j,s)) + δ(δ(i,s),j)) : nothing
s > 2 ? input.H .*= J : input.H .*= J/2 

input.rsite = ones(Float64, s) .* 100.0

@tullio input.dipGE[n,x] := δ(1,x) (n in 1:s, x in 1:3)

Model = createModel() # creates an empty Model class 
initializeModel!(Model, input) # set up the Model according to the input

# ### Example of absorption spectrum simulation
abs_response_function = ensambleDiagram(getAbsResp, Model, parallel = "n", verbose = 1); # "verbose" to 1 to see console output
freq, absorption = processDiagram(abs_response_function, Model, conv = 1000.0, norm = 1) # "conv" convolutes spectrum with Gaussian function with std = 1000 fs; "norm" normalizes spectra to 1
p1 = plot(freq, absorption, xlims = (-100, 1000), label = "Absorption") 

### Example of fluorescence spectrum simulation
flor_response_function = ensambleDiagram(getTrfResp, Model, trfLES = "optim", parallel = "n", verbose = 1); # "trfLES" chooses the method to find the lowest energy state of the excited state  
freq, fluorescence = processDiagram(flor_response_function, Model, conv = 1000.0, norm = 1)
p1 = plot!(freq, fluorescence, xlims = (-1000, 1000), label = "Fluorescence") 

# ### Example of 2DES speca simulation

# # Simulation of rephasing and non-rephasing diagrams of GSB, ESA and ESE.
GSB_R = ensembleDiagram2D(getDiagram2D, "GSB", "R", Model, parallel = "n", verbose = 1);
ESE_R = ensembleDiagram2D(getDiagram2D, "ESE", "R", Model, parallel = "n", verbose = 1);
ESA_R = ensembleDiagram2D(getDiagram2D, "ESA", "R", Model, parallel = "n", verbose = 1);
GSB_nR = ensembleDiagram2D(getDiagram2D, "GSB", "NR", Model, parallel = "n", verbose = 1);
ESE_nR = ensembleDiagram2D(getDiagram2D, "ESE", "NR", Model, parallel = "n", verbose = 1);
ESA_nR = ensembleDiagram2D(getDiagram2D, "ESA", "NR", Model, parallel = "n", verbose = 1);

# Making rephasing and non-rephasing signals
diag_R  = GSB_R  +ESE_R  +ESA_R;
diag_nR = GSB_nR +ESE_nR +ESA_nR;

# Plotting parameters
zmax = 1.0
levels = 21
@unpack time2D = Model.pr

# Plotting signals of diagrams and 
hGSB_R = plot2D(GSB_R, "R", zmax, levels, time2D, conv = 500.0, title = "R GSB")
hESE_R = plot2D(ESE_R, "R", zmax, levels, time2D, conv = 500.0, title = "R ESE")
hESA_R = plot2D(ESA_R, "R", zmax, levels, time2D, conv = 500.0, title = "R ESA")
h_R    = plot2D(diag_R, "R", zmax, levels, time2D, conv = 500.0, title = "Rephasing")

hGSB_nR = plot2D(GSB_nR, "nR", zmax, levels, time2D, conv = 500.0, title = "nR GSB")
hESE_nR = plot2D(ESE_nR, "nR", zmax, levels, time2D, conv = 500.0, title = "nR ESE")
hESA_nR = plot2D(ESA_nR, "nR", zmax, levels, time2D, conv = 500.0, title = "nR ESA")
h_nR    = plot2D(diag_nR, "nR", zmax, levels, time2D, conv = 500.0, title = "Non-Rephasing")   
