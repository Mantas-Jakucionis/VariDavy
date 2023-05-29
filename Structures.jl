const cm2fs = 0.000188365156731
const fs2cm = 5308.837458872913
const kbcm = 0.69503476
const kbfs = 0.000130920331500893

# Spectral density fuctions in units of [cm].
function Drude(w, p)
    g = p[1]*cm2fs # - damping constant
    return w/(w^2 + g^2)
end 
 
function Ohmic(w, p) 
    a  = p[1] # - order 
    wc = p[2]*cm2fs # - cut-off frequency
    return 1 / w * (w/wc)^(a-1) * exp(-w/wc)
end

function LogNormal(w, p)
    g  = p[1] # - sigma
    wc = p[2]*cm2fs # - cut-off frequency
    s  = p[3] # - Huang-Rhys factor
    return s / g / w * exp( - log(w/wc)^2 / 2 / g^2 ) / sqrt(2*pi)
end

function WSCP_4mode(w, _)
    g  = [0.4, 0.2, 0.2, 0.4]
    wc = [28, 54, 90, 145]
    s  = [0.45, 0.15, 0.21, 0.15]
    return sum( map( k -> LogNormal(w, [g[k], wc[k], s[k]]), 1:length(g) ) )
end

function WSCP_3mode(w, _)
    g  = [0.4, 0.25, 0.2]
    wc = [26, 51, 85]
    s  = [0.39, 0.23, 0.23]
    return sum( map( k -> LogNormal(w, [g[k], wc[k], s[k]]), 1:length(g) ) )
end

function SP(w, _)
    g  = [0.47]
    wc = [35.0]
    s  = [1.7]
    return sum( map( k -> LogNormal(w, [g[k], wc[k], s[k]]), 1:length(g) ) )
end

function B777(w, _)
    S = 0.8
    si = [0.8, 0.5]
    wi = [0.556527, 1.935747] .* cm2fs
    A = S/sum(si)
    B = map( k -> si[k] * w^3 * exp(-(w/wi[k])^0.5) / (10080*wi[k]^4), 1:length(si) )
    return A*sum(B)
end

@with_kw mutable struct dimensionStruct
    nmultp0M::Int = -1
    nmultp1M::Int = -1
    nmultp2M::Int = -1

    nsite0M::Int = -1
    nsite1M::Int  = -1
    nsite2M::Int = -1

    nbath0M::Int = -1
    nmode0M::Int = -1

    nbath1M::Int = -1
    nmode1M::Int = -1

    nbath2M::Int = -1
    nmode2M::Int = -1

    ntraj::Int = -1
    ntime::Int = -1

    ntrialsLES::Int = -1

    statesize1M::Int = -1
    astatesize1M::Int = -1
    lstatesize1M::Int = -1

    statesize0M::Int = -1
    astatesize0M::Int = -1
    lstatesize0M::Int = -1

    statesize2M::Int = -1
    astatesize2M::Int = -1
    lstatesize2M::Int = -1
end

@with_kw mutable struct parameterStruct
    atol::Float64 = -1
    rtol::Float64 = -1
    iterRtol::Float64 = -1
    apopThres::Float64 = -1
    d2Thres::Float64 = -1
    singleMultipleLES::Int = -1

    thermalization::Int = -1
    thermTempOffset::Float64 = -1
    thermMethod::String = "-1"
    thermType::String = "-1"
    thermStep::Float64 = -1
    thermRate::Float64 = -1
    thermProb::Float64 = -1

    trfThermTime::Float64 = -1
    
    multpDistr::String = "-1"
    multpPerLayer::Int = -1
    multpDistance::Float64 = -1

    propagateSimply::Int = -1
    propagationMethod = nothing

    particleType::String = "-1"

    H::Array{Float64} = Array{Float64}(undef)
    K::Array{Float64} = Array{Float64}(undef)
    rsite::Array{Float64} = Array{Float64}(undef)
    dipGE::Array{Float64} = Array{Float64}(undef)
    iwG::Array{Float64} = Array{Float64}(undef)
    iwE::Array{Float64} = Array{Float64}(undef)
    id::Array{Float64} = Array{Float64}(undef)
    f::Array{Float64} = Array{Float64}(undef)
    diw::Array{Float64} = Array{Float64}(undef)
    anharmQ::Bool = false

    bathSpectralDensity = nothing
    bathSpdParameters = Array{Float64}(undef)
    staticDisorderSigma::Float64 = -1
    disordered::Int = -1

    extFieldDistribution::String = "-1"
    extEstatic::Array{Float64} = Array{Float64}(undef)
    
    tmin::Float64 = -1
    tmax::Float64 = -1
    dt::Float64 = -1
    time::Array{Float64} = Array{Float64}(undef)
    
    thermTimes::Array{Float64} = Array{Float64}(undef)

    temp::Float64 = -1
    wmin::Float64 = -1
    wmax::Float64 = -1
    dw::Float64 = -1
    dwp::Float64 = -1

    twaiting2D::Float64 = -1
    tmin2D::Float64 = -1
    tmax2D::Float64 = -1
    dt2D::Float64 = -1
    time2D::Array{Float64} = Array{Float64}(undef)
end

@with_kw mutable struct indexStruct

    Dmt::Array{Int} = Array{Int}(undef)
    Dn::Array{Int} = Array{Int}(undef)
    Dk::Array{Int} = Array{Int}(undef)
    Dm::Array{Int} = Array{Int}(undef)

    Dmt0M::Array{Int} = Array{Int}(undef)
    Dn0M::Array{Int} = Array{Int}(undef)
    Dn2M::Array{Int} = Array{Int}(undef)

    ai1M::Array{Int} = Array{Int}(undef)
    li1M::Array{Int} = Array{Int}(undef)

    ai1M_MTP1::Array{Int} = Array{Int}(undef)
    li1M_MTP1::Array{Int} = Array{Int}(undef)

    ai0M::Array{Int} = Array{Int}(undef)
    li0M::Array{Int} = Array{Int}(undef)

    ai2M::Array{Int} = Array{Int}(undef)
    li2M::Array{Int} = Array{Int}(undef)

    aRi1M::Array{Int} = Array{Int}(undef)
    aIi1M::Array{Int} = Array{Int}(undef)
    lRi1M::Array{Int} = Array{Int}(undef)
    lIi1M::Array{Int} = Array{Int}(undef)

    aRi0M::Array{Int} = Array{Int}(undef)
    aIi0M::Array{Int} = Array{Int}(undef)
    lRi0M::Array{Int} = Array{Int}(undef)
    lIi0M::Array{Int} = Array{Int}(undef)

    aRi2M::Array{Int} = Array{Int}(undef)
    aIi2M::Array{Int} = Array{Int}(undef)
    lRi2M::Array{Int} = Array{Int}(undef)
    lIi2M::Array{Int} = Array{Int}(undef)

    i21::Array{Tuple{Int, Int}} = Array{Tuple{Int, Int}}(undef)
    i12::Array{Int, 2} = Array{Int, 2}(undef, 2, 2)

    N1::Array{Float64} = Array{Float64}(undef)
    N2::Array{Float64} = Array{Float64}(undef)

end

@with_kw struct thermStruct
    times::Vector{Float64}
    events::Array{Int, 2}
    thermStruct(t::Vector{Float64}, e::Array{Int, 2}) = new(t, e)
    thermStruct(a::Int) = new(Array{Float64, 1}(), zeros(Int, 1, 1))
end

@with_kw mutable struct stateStruct

    manifold::Int
    initialState::Vector{ComplexF64}
    tinitial::Float64
    tfinal::Float64
    norm::Float64
    propagateSimply::Int
    dyn::Union{ODESolution, Nothing}

    stateStruct(manifold::Int, iState::Vector{ComplexF64}, iTime::Float64, propSimply::Int) = 
                new(manifold, iState, iTime, iTime, 1.0, propSimply, nothing)
    stateStruct(t::Float64, s::stateStruct) = 
                new(s.manifold, s.dyn(t), t, t, -1.0, s.propagateSimply, nothing)
    stateStruct(sVec::Vector{ComplexF64}, s::stateStruct) = 
                new(s.manifold, sVec, t, t, -1.0, s.propagateSimply, nothing)
    stateStruct(manifold::Int, iState::Vector{ComplexF64}, iTime::Float64, propSimply::Int) = 
                new(manifold, iState, iTime, iTime, 1.0, propSimply, nothing)
end

@with_kw mutable struct ensembleStruct
    u0::Vector{Vector{ComplexF64}} = Vector{Vector{ComplexF64}}()
    eVec::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    dE::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    therm::Vector{thermStruct} = Vector{thermStruct}()
    thermTrf::Vector{thermStruct} = Vector{thermStruct}()
    lowestEnergyState1M::stateStruct = stateStruct(-1, [ComplexF64(0.0, 0.0)], 0.0, -1)
end

@with_kw mutable struct trajectoryStruct
    u0::Vector{ComplexF64} = Vector{ComplexF64}()
    eVec::Vector{Float64} = Vector{Float64}()
    dE::Vector{Float64} = Vector{Float64}()
    therm::thermStruct = thermStruct(0)
    thermTrf::thermStruct = thermStruct(0)
    lowestEnergyState1M::stateStruct = stateStruct(-1, [ComplexF64(0.0, 0.0)], 0.0, -1)
end

@with_kw struct modelStruct
    nd::dimensionStruct
    id::indexStruct
    pr::parameterStruct
    ens::Union{ensembleStruct, Nothing}
    traj::Union{trajectoryStruct, Nothing}
    modelStruct(a::dimensionStruct,b::indexStruct,c::parameterStruct,ens::ensembleStruct) = 
                new(a,b,c,ens,nothing)
    modelStruct(m::modelStruct, traj::trajectoryStruct) = 
                new(m.nd, m.id, deepcopy(m.pr), nothing, traj)
end

@with_kw struct cachePackStructFloat64
    model::modelStruct
    M::Array{Float64, 2}
    RHS::Vector{ComplexF64}
end

@with_kw struct cachePackStructComplexF64
    model::modelStruct
    M::Array{ComplexF64, 2}
    RHS::Vector{ComplexF64} 
end

@with_kw struct transitionDipoleOperator
    polarizationVector::Array{Float64}
    action::Int
    dir::String
    transitionDipoleOperator(vec::Array{Float64}, action::Int) = new(vec, action, "R")
    transitionDipoleOperator(vec::Array{Float64}, action::Int, dir::String) = new(vec, action, dir)
end

@with_kw struct diffStruct
    E0::Float64
    dE::Float64
    model::modelStruct
end
