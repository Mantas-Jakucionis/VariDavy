using Plots; default(size=(1200,800), thickness_scaling=2)
using FFTW
using DelimitedFiles
using Distributed

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

### An example model of homodimer coupled to 1 intramolecular vibrational mode per molecule at 77 temperature. 
### Using mD2 ansatz (nmultp = 5), averaging over 30 wavefunction trajectories
input = Example_Model(

                    nmultp = 5,                    
                    nsite = 2,
                    ntraj = 30,

                    H = [0.0 -100
                        -100  0.0],

                    dipGE = [1.0 0.0 0.0
                             1.0 0.0 0.0], # identical transition dipoles 
                    
                    iwG = [500.0
                           500.0],

                    iwE = [500.0
                           500.0], # identical QHO frequencies in both ground and excited states

                    id =  [sqrt(2)
                           sqrt(2)], # Equal to HR factor = 1

                    tmax = 2000.0,
                                    
                    wmax = 0.0, # removes bath modes and uses only the iwG and iwE modes
                                        
                    temp = 77,

                    twaiting2D = 0.0,
                    tmax2D     = 300.0,
                    dt2D       = 5.0

                    )

Model = createModel()
initializeModel!(Model, input)

### Example of absorption spectrum simulation
abs_response_function = ensambleDiagram(getAbsResp, Model, parallel = "t", verbose = 1); # "t" parallelizes simulation of wavefunction trajectories over all available CPU threads
freq, absorption = processDiagram(abs_response_function, Model, conv = 500.0, norm = 1) 
p1 = plot!(freq, absorption, xlims = (-1500, 2000), label = "Absorption") 

### Example of fluorescence spectrum simulation
flor_response_function = ensambleDiagram(getTrfResp, Model, trfLES = "optim", parallel = "t", verbose = 1); # "trfLES" chooses the method to find the lowest energy state of the excited state 
freq, fluorescence = processDiagram(flor_response_function, Model, conv = 500.0, norm = 1)
p1 = plot!(freq, fluorescence, xlims = (-4500, 3000), label = "Fluorescence") 

### Example of 2DES speca simulation

### Simulation of rephasing and non-rephasing diagrams of GSB, ESA and ESE.
### Might take a while
GSB_R = ensembleDiagram2D( getDiagram2D, "GSB", "R",  Model, parallel = "t", verbose = 1);
ESE_R = ensembleDiagram2D( getDiagram2D, "ESE", "R",  Model, parallel = "t", verbose = 1);
ESA_R = ensembleDiagram2D( getDiagram2D, "ESA", "R",  Model, parallel = "t", verbose = 1);
GSB_nR = ensembleDiagram2D(getDiagram2D, "GSB", "NR", Model, parallel = "t", verbose = 1);
ESE_nR = ensembleDiagram2D(getDiagram2D, "ESE", "NR", Model, parallel = "t", verbose = 1);
ESA_nR = ensembleDiagram2D(getDiagram2D, "ESA", "NR", Model, parallel = "t", verbose = 1);

# Making rephasing and non-rephasing signals
diag_R  = GSB_R  +ESE_R  +ESA_R;
diag_nR = GSB_nR +ESE_nR +ESA_nR;

# Plotting parameters
zmax = 1.0
levels = 21
@unpack time2D = Model.pr

# Plotting signals of diagrams 
hGSB_R = plot2D( GSB_R, "R", zmax, levels, time2D, conv = 500.0, title = "R GSB")
hESE_R = plot2D( ESE_R, "R", zmax, levels, time2D, conv = 500.0, title = "R ESE")
hESA_R = plot2D( ESA_R, "R", zmax, levels, time2D, conv = 500.0, title = "R ESA")
h_R    = plot2D(diag_R, "R", zmax, levels, time2D, conv = 500.0, title = "Rephasing")

hGSB_nR = plot2D( GSB_nR, "nR", zmax, levels, time2D, conv = 500.0, title = "nR GSB")
hESE_nR = plot2D( ESE_nR, "nR", zmax, levels, time2D, conv = 500.0, title = "nR ESE")
hESA_nR = plot2D( ESA_nR, "nR", zmax, levels, time2D, conv = 500.0, title = "nR ESA")
h_nR    = plot2D(diag_nR, "nR", zmax, levels, time2D, conv = 500.0, title = "Non-Rephasing")   

ES2D    = plot2D(diag_R, diag_nR, "2D", zmax, levels, time2D, conv = 500.0, title = "2DES")

