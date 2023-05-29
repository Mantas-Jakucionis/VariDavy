function getStateDynamicsProbabilities(state::stateStruct, model::modelStruct)
    @unpack nd, pr = model
    time = collect(state.tinitial:pr.dt:state.tfinal)
    xt  = map(state.dyn, time)
    ca  = map(x-> getAlpha(x, state.manifold, model), xt)
    cl = map(x-> getLambda(x, state.manifold, model), xt)
    S   = map(getS, cl, cl)
    nsite = -1
    if state.manifold == 0
        nsite = nd.nsite0M
    elseif state.manifold == 1
        nsite = nd.nsite1M
    elseif state.manifold == 2
        nsite = nd.nsite2M
    end

    prob = real.(map(n -> begin @tullio  prob[t] := conj(ca[t][i,$n])*ca[t][j,$n]*S[t][i,j] end, 1:nsite))
    norm = sum(prob)
    prob = map(n -> prob[n] ./ norm, 1:nsite)
    return time, prob
end

function getStateDynamicsExcitonProbabilities(state::stateStruct, model::modelStruct)
    @unpack nd, pr = model
    time = collect(state.tinitial:pr.dt:state.tfinal)
    _, excVec = eigen(model.pr.H)
    xt  = map(state.dyn, time)
    ca  = map(x-> getAlpha(x, state.manifold, model), xt)
    cl = map(x-> getLambda(x, state.manifold, model), xt)
    S   = map(getS, cl, cl)

    nexc = -1
    if state.manifold == 0
        nexc = nd.nsite0M
    elseif state.manifold == 1
        nexc = nd.nsite1M
    elseif state.manifold == 2
        nexc = nd.nsite2M
    end

    prob = real.(map(k -> begin @tullio  prob[t] := excVec'[$k,n]*conj(ca[t][i,n])*ca[t][j,m]*excVec[m,$k]*S[t][i,j] end, 1:nexc))
    norm = sum(prob)
    prob = map(n -> prob[n] ./ norm, 1:nexc)
    return time, prob
end

function getStateDynamicsDensityMatrix(state::stateStruct, model::modelStruct)
    @unpack pr = model
    time = collect(state.tinitial:pr.dt:state.tfinal)
    xt  = map(state.dyn, time)
    ca  = map(x-> getAlpha(x, state.manifold, model), xt)
    cl = map(x-> getLambda(x, state.manifold, model), xt)
    S   = map(getS, cl, cl)
    @tullio  densityMatrix[t,n,m] := conj(ca[t][i,n])*ca[t][j,m]*S[t][i,j]
    @tullio  norm[t] := densityMatrix[t,n,n]
    @tullio  densityMatrixNorm[t,n,m] := densityMatrix[t,n,m] / norm[t]
    return time, densityMatrixNorm
end

function getStateDynamicsExcitonDensityMatrix(state::stateStruct, model::modelStruct)
    @unpack pr = model
    time = collect(state.tinitial:pr.dt:state.tfinal)
    _, excVec = eigen(model.pr.H)
    xt  = map(state.dyn, time)
    ca  = map(x-> getAlpha(x, state.manifold, model), xt)
    cl = map(x-> getLambda(x, state.manifold, model), xt)
    S   = map(getS, cl, cl)
    @tullio  densityMatrix[t,k,l] := excVec'[k,n]*conj(ca[t][i,n])*ca[t][j,m]*excVec[m,l]*S[t][i,j]
    @tullio  norm[t] := densityMatrix[t,k,k]
    @tullio  densityMatrixNorm[t,k,l] := densityMatrix[t,k,l] / norm[t]
    return time, densityMatrixNorm
end

function getVibrationsQuantaNumber(state::stateStruct, model::modelStruct)
    @unpack nd, pr = model
    time = collect(state.tinitial:pr.dt:state.tfinal)
    xt  = map(state.dyn, time)
    ca  = map(x-> getAlpha(x, state.manifold, model), xt)
    cl = map(x-> getLambda(x, state.manifold, model), xt)
    S   = map(getS, cl, cl)

    nsite = -1
    if state.manifold == 0
        nsite = nd.nsite0M
    elseif state.manifold == 1
        nsite = nd.nsite1M
    elseif state.manifold == 2
        nsite = nd.nsite2M
    end

    prob = real.(map(n -> begin @tullio  prob[t] := conj(ca[t][i,$n])*ca[t][j,$n]*S[t][i,j] end, 1:nsite))
    norm = sum(prob)

    @tullio  quanta[t,n,q] := conj(ca[t][i,m])*ca[t][j,m]*S[t][i,j]*( conj(cl[t][i,n,q]) * cl[t][j,n,q] ) / norm[t]
    return real(quanta)
end

function getVibrationsCoordinateVariance(state::stateStruct, model::modelStruct)
    @unpack nd, pr = model
    time = collect(state.tinitial:pr.dt:state.tfinal)
    xt  = map(state.dyn, time)
    ca  = map(x-> getAlpha(x, state.manifold, model), xt)
    cl = map(x-> getLambda(x, state.manifold, model), xt)
    S   = map(getS, cl, cl)

    nsite = -1
    if state.manifold == 0
        nsite = nd.nsite0M
    elseif state.manifold == 1
        nsite = nd.nsite1M
    elseif state.manifold == 2
        nsite = nd.nsite2M
    end

    prob = real.(map(n -> begin @tullio  prob[t] := conj(ca[t][i,$n])*ca[t][j,$n]*S[t][i,j] end, 1:nsite))
    norm = sum(prob)

    @tullio   x[t,n,q] := conj(ca[t][i,m])*ca[t][j,m]*S[t][i,j]*( conj(cl[t][i,n,q]) + cl[t][j,n,q] ) / sqrt(2.0) / norm[t]
    @tullio  x2[t,n,q] := conj(ca[t][i,m])*ca[t][j,m]*S[t][i,j]*( 1 + (conj(cl[t][i,n,q]))^2 + (cl[t][j,n,q])^2 + 2*conj(cl[t][i,n,q])*cl[t][j,n,q] ) / 2.0 / norm[t]

    return real.(x2 .- x.^2)
end

function getVibrationsMomentumVariance(state::stateStruct, model::modelStruct)
    @unpack nd, pr = model
    time = collect(state.tinitial:pr.dt:state.tfinal)
    xt  = map(state.dyn, time)
    ca  = map(x-> getAlpha(x, state.manifold, model), xt)
    cl = map(x-> getLambda(x, state.manifold, model), xt)
    S   = map(getS, cl, cl)

    nsite = -1
    if state.manifold == 0
        nsite = nd.nsite0M
    elseif state.manifold == 1
        nsite = nd.nsite1M
    elseif state.manifold == 2
        nsite = nd.nsite2M
    end

    prob = real.(map(n -> begin @tullio  prob[t] := conj(ca[t][i,$n])*ca[t][j,$n]*S[t][i,j] end, 1:nsite))
    norm = sum(prob)

    @tullio   p[t,n,q] := im*conj(ca[t][i,m])*ca[t][j,m]*S[t][i,j]*( conj(cl[t][i,n,q]) - cl[t][j,n,q] ) / sqrt(2.0) / norm[t]
    @tullio  p2[t,n,q] :=  -conj(ca[t][i,m])*ca[t][j,m]*S[t][i,j]*( -1 + (conj(cl[t][i,n,q]))^2 + (cl[t][j,n,q])^2 - 2*conj(cl[t][i,n,q])*cl[t][j,n,q] ) / 2.0  / norm[t]

    return real.(p2 .- p.^2)
end

function getVibrationsCoordinate(state::stateStruct, model::modelStruct)
    @unpack nd, pr = model
    time = collect(state.tinitial:pr.dt:state.tfinal)
    xt  = map(state.dyn, time)
    ca  = map(x-> getAlpha(x, state.manifold, model), xt)
    cl = map(x-> getLambda(x, state.manifold, model), xt)
    S   = map(getS, cl, cl)

    nsite = -1
    if state.manifold == 0
        nsite = nd.nsite0M
    elseif state.manifold == 1
        nsite = nd.nsite1M
    elseif state.manifold == 2
        nsite = nd.nsite2M
    end

    prob = real.(map(n -> begin @tullio  prob[t] := conj(ca[t][i,$n])*ca[t][j,$n]*S[t][i,j] end, 1:nsite))
    norm = sum(prob)

    @tullio   x[t,n,q] := real <| conj(ca[t][i,m])*ca[t][j,m]*S[t][i,j]*( conj(cl[t][i,n,q]) + cl[t][j,n,q] ) / sqrt(2.0) / norm[t]
    return x 
end

function getVibrationsMomentum(state::stateStruct, model::modelStruct)
    @unpack nd, pr = model
    time = collect(state.tinitial:pr.dt:state.tfinal)
    xt  = map(state.dyn, time)
    ca  = map(x-> getAlpha(x, state.manifold, model), xt)
    cl = map(x-> getLambda(x, state.manifold, model), xt)
    S   = map(getS, cl, cl)

    nsite = -1
    if state.manifold == 0
        nsite = nd.nsite0M
    elseif state.manifold == 1
        nsite = nd.nsite1M
    elseif state.manifold == 2
        nsite = nd.nsite2M
    end

    prob = real.(map(n -> begin @tullio  prob[t] := conj(ca[t][i,$n])*ca[t][j,$n]*S[t][i,j] end, 1:nsite))
    norm = sum(prob)

    @tullio   p[t,n,q] := real <| im*conj(ca[t][i,m])*ca[t][j,m]*S[t][i,j]*( conj(cl[t][i,n,q]) - cl[t][j,n,q] ) / sqrt(2.0) / norm[t]
    return p
end

function ensambleDiagram(fun, model::modelStruct; parallel::String = "n", verbose::Int = 1, trfLES::String = "prop-therm", saveLESEnergies::Bool = false, saveLESStateDyn::Bool = false)

    args = Dict("trfLES" => trfLES, 
                "saveLESEnergies" => saveLESEnergies,
                "saveLESStateDyn" => saveLESStateDyn)

    saveDataQ = any((saveLESEnergies))

    if fun == getTrfResp

        if trfLES == "optim"
            if model.ens.lowestEnergyState1M.manifold == -1 && model.pr.disordered == 0
                F = Bool(model.pr.singleMultipleLES) ? getLowestEnergyState1M_MTP1 : getLowestEnergyState1M
                model.ens.lowestEnergyState1M = F(model, 1, ntrials = model.nd.ntrialsLES)
            end
        elseif trfLES == "exc"
            nothing
        elseif trfLES == "prop-therm"
            nothing
        elseif trfLES == "prop"
            nothing
        elseif trfLES == "prop-therm-ergo"
            if model.ens.lowestEnergyState1M.manifold == -1 && model.pr.disordered == 0

                tmodel = getTrajectoryModel(1, model, recomp=0)

                @unpack pr, traj = tmodel
                @unpack u0, eVec = traj
            
                state = createState(0, u0, 0.0, 1)
                k = transitionDipoleOperator(eVec, +1)
                state = getNewState(k, state, tmodel)
                propagate!(state, pr.trfThermTime, tmodel)
                model.ens.lowestEnergyState1M = deepcopy(state)
            end

        else
            error("Unknown 'trfLES' variable value.")
        end

    end

    if parallel == "n"

        out = map(k -> begin
            tcalc = @elapsed x, auxData = fun(model, k, args=args)
            verbose == 1 ? println("w:",Distributed.myid()-1," ","t:",Threads.threadid()," ", ": Trajectory ", k, " is done (", round(tcalc, digits=2), " s).") : nothing
            return x, auxData
        end, 1:model.nd.ntraj)

        trajLinResp = getindex.(out, 1)
        auxData = reduce(vcat, getindex.(out, 2))

        return saveDataQ ? (sum(trajLinResp) ./ model.nd.ntraj, auxData) : (sum(trajLinResp) ./ model.nd.ntraj)

    elseif parallel == "t"

        trajLinResp = Array{Vector{ComplexF64}}(undef, model.nd.ntraj)
        auxData = Array{Dict{String, Any}}(undef, model.nd.ntraj)

        Threads.@threads for k in 1:model.nd.ntraj
            tcalc = @elapsed trajLinResp[k], auxData[k] = fun(model, k, args=args)            
            verbose == 1 ? println("w:",Distributed.myid()-1," ","t:",Threads.threadid()," ", ": Trajectory ", k, " is done (", round(tcalc, digits=2), " s).") : nothing
        end

        return saveDataQ ? (sum(trajLinResp) ./ model.nd.ntraj, auxData) : (sum(trajLinResp) ./ model.nd.ntraj)

    elseif parallel == "p"

        out = pmap(k -> begin
            tcalc = @elapsed x, auxData = fun(model, k, args=args)
            verbose == 1 ? println("w:",Distributed.myid()-1," ","t:",Threads.threadid()," ", ": Trajectory ", k, " is done (", round(tcalc, digits=2), " s).") : nothing
            return x, auxData
        end, 1:model.nd.ntraj);

        trajLinResp = getindex.(out, 1)
        auxData = reduce(vcat, getindex.(out, 2))

        return saveDataQ ? (sum(trajLinResp) ./ model.nd.ntraj, auxData) : sum(trajLinResp) ./ model.nd.ntraj

    elseif parallel == "pt"

        nthreads = Threads.nthreads()
        nworkers = Distributed.nprocs() - 1

        ntraj = model.nd.ntraj
        ntraj_distr = collect(Iterators.partition(1:ntraj, ceil.(Int, ntraj/nworkers)))

        ntraj_threads = collect.(Iterators.partition.(ntraj_distr, nthreads)) 
        ntraj_threads = reduce(vcat, ntraj_threads)

        n_pmap = length(ntraj_threads)

        out = pmap(k -> begin
        
            xlen = length(ntraj_threads[k])
            x = Array{Vector{ComplexF64}}(undef, xlen)
            auxData = Array{Dict{String, Any}}(undef, xlen)

            Threads.@threads for i in 1:xlen
                tcalc = @elapsed x[i], auxData[i] = fun(model, ntraj_threads[k][i], args=args)    
                verbose == 1 ? println("w:",Distributed.myid()-1," ","t:",Threads.threadid()," ", ": Trajectory ", ntraj_threads[k][i], " is done (", round(tcalc, digits=2), " s).") : nothing
            end

            return saveDataQ ? (sum(x), auxData) : sum(x)
            
        end, 1:n_pmap);

        if saveDataQ
            trajLinResp = getindex.(out, 1)
            auxData = reduce(vcat, getindex.(out, 2))
        else
            trajLinResp = out
        end

        return saveDataQ ? (sum(trajLinResp) ./ model.nd.ntraj, auxData) : sum(trajLinResp) ./ model.nd.ntraj

    else
        error("Unknown parallelelization method.")
    end
end

function getTrfResp(model::modelStruct, k::Int = 1; args::Dict{String, Any})
    trfLES = get(args, "trfLES", "WARNING: TRF response requires trfLES solution argument.")
    saveLESEnergies = get(args, "saveLESEnergies", "WARNING: Variable saveLESEnergies value is invalid.")
    saveLESStateDyn = get(args, "saveLESStateDyn", "WARNING: Variable saveLESStateDyn value is invalid.")

    recompQ = (trfLES == "optim" && model.pr.disordered == 1) ? 1 : 0
    
    tmodel = getTrajectoryModel(k, model; recomp = recompQ)
    return getTrfRespViaField(tmodel, trfLES, saveLESEnergies, saveLESStateDyn)
end

function getTrfRespViaField(tmodel::modelStruct, trfLES::String, saveLESEnergies::Bool, saveLESStateDyn::Bool)

    @unpack id, pr, nd, traj = tmodel
    @unpack temp, iwE = pr
    
    u0 = traj.u0
    eVec = traj.eVec
    tT = pr.trfThermTime

    if trfLES == "optim" || trfLES == "exc" 

        refState = deepcopy(tmodel.traj.lowestEnergyState1M)
        if temp != 0.0 
            E0 = getStateVectorEnergy1M(refState.initialState, tmodel)
            @tullio  dE  = pr.iwE[n,h]*abs2(getModeDisplacement(pr.iwE[n,h], temp, tmodel))
            refState.initialState = getThemalFluctuationStateVector1M(refState, E0, dE, tmodel)
        end

        saveLESEnergies ? initialEnergy = getStateVectorEnergy1M(refState.initialState, tmodel, normAmp=fs2cm) : nothing
        saveLESEnergies ? finalEnergy = initialEnergy : nothing

        saveLESStateDyn ? saveState = deepcopy(refState) : nothing 

    elseif trfLES == "prop" 
                
        refState = createState(0, u0, 0.0, 1)
        k1 = transitionDipoleOperator(eVec, +1)
        refState = getNewState(k1, refState, tmodel)
        saveLESEnergies ? initialEnergy = getStateVectorEnergy1M(refState.initialState, tmodel, normAmp=fs2cm) : nothing
        propagate!(refState, tT, tmodel)
        saveLESEnergies ? finalEnergy = getStateVectorEnergy1M(refState.dyn(tT), tmodel, normAmp=fs2cm) : nothing
        saveLESStateDyn ? saveState = deepcopy(refState) : nothing
        refState = createStateAt(tT, refState, tmodel)
        refState.tinitial = 0.0
        refState.tfinal = 0.0

    elseif trfLES == "prop-therm" 
                
        refState = createState(0, u0, 0.0, 1)
        k1 = transitionDipoleOperator(eVec, +1)
        refState = getNewState(k1, refState, tmodel)
        saveLESEnergies ? initialEnergy = getStateVectorEnergy1M(refState.initialState, tmodel, normAmp=fs2cm) : nothing
        propagate!(refState, tT, tmodel, therm = "trf")
        saveLESEnergies ? finalEnergy = getStateVectorEnergy1M(refState.dyn(tT), tmodel, normAmp=fs2cm) : nothing
        saveLESStateDyn ? saveState = deepcopy(refState) : nothing
        refState = createStateAt(tT, refState, tmodel)
        refState.tinitial = 0.0
        refState.tfinal = 0.0
          
    else
        error("TRF response requires trfLES solution argument.")
    end

    cl = getLambda(refState.initialState, 1, tmodel)
    S  = getS(cl, cl)
    
    R = deepcopy(refState)
    k1 = transitionDipoleOperator(eVec, -1)
    R = getNewState(k1, R, tmodel)    
    
    L = deepcopy(refState) 
    if tmodel.nd.nmultp1M > 1 
        if all(abs.(S)[1,2:end] .< pr.d2Thres)
            R.propagateSimply = 3
            setUnpopulatedMultiples!(L.initialState, 1, tmodel)
        end 
    end
    propagate!(R, pr.tmax, tmodel)
    propagate!(L, pr.tmax, tmodel)
    kI = transitionDipoleOperator(eVec, -1, "L")
    
    LESenergies = saveLESEnergies ?  [initialEnergy, finalEnergy] : nothing
    saveState = saveLESStateDyn ? saveState : nothing
    auxData = Dict("LESenergies" => LESenergies, "saveState" => saveState)

    return (map(t -> expValueAt(t, L, kI, R, tmodel), tmodel.pr.time), auxData)

end

function getLowestEnergyState1M(model::modelStruct, k::Int=1; ntrials::Int=1)

    @unpack id, pr, nd = model

    if !isnothing(model.traj)
        @unpack u0, eVec = model.traj 
    elseif !isnothing(model.ens)
        u0 = model.ens.u0[k]
        eVec = model.ens.eVec[k]
    else
        error("Model is uninitialized.")
    end

    trialState = Vector{stateStruct}(undef, ntrials)
    trialStateEnergies = Vector{Float64}(undef, ntrials)
    dipU = transitionDipoleOperator(eVec, +1)

    trialOptimTime = Vector{Float64}(undef, ntrials)

    Threads.@threads for i in 1:ntrials
        
        trialState[i] = createState(0, u0, 0.0, 1)
        trialState[i] = getNewState(dipU, trialState[i], model)
        
        trialState[i].initialState[id.li1M[1,:,:]] .= pr.f
        setUnpopulatedMultiples!(trialState[i].initialState, 1, model)

        trialOptimTime[i] = @elapsed trialState[i].initialState = findMinimalEnergyStateVector1M(trialState[i], model)
        trialState[i].initialState = getNormalizedStateVector1M(trialState[i].initialState, model, trialState[i].norm)
        
        trialStateEnergies[i] = getStateVectorEnergy1M(trialState[i].initialState, model, normAmp=fs2cm)

    end
    
    minIndex = findmin(trialStateEnergies)[2]
    
    println("Lowest energy 1M state was found: E=$(trialStateEnergies[minIndex])")
    println("LEES optimization time: $(trialOptimTime[minIndex])")

    return trialState[minIndex] 
end

function getLowestEnergyState1M_MTP1(model::modelStruct, k::Int=1; ntrials::Int=1)

    @unpack id, pr, nd = model

    if !isnothing(model.traj)
        @unpack u0, eVec = model.traj 
    elseif !isnothing(model.ens)
        u0 = model.ens.u0[k]
        eVec = model.ens.eVec[k]
    else
        error("Model is uninitialized.")
    end

    trialState = Vector{stateStruct}(undef, ntrials)
    trialStateEnergies = Vector{Float64}(undef, ntrials)
    dipU = transitionDipoleOperator(eVec, +1)
    
    trialOptimTime = Vector{Float64}(undef, ntrials)

    Threads.@threads for i in 1:ntrials
        
        trialState[i] = createState(0, u0, 0.0, 1)
        trialState[i] = getNewState(dipU, trialState[i], model)
        
        trialState[i].initialState[id.li1M[1,:,:]] .= pr.f
        
        setUnpopulatedMultiples!(trialState[i].initialState, 1, model)
        
        trialOptimTime[i] = @elapsed trialState[i].initialState = findMinimalEnergyStateVector1M_MTP1(trialState[i], model) 
        trialState[i].initialState = getNormalizedStateVector1M(trialState[i].initialState, model, trialState[i].norm)
        
        trialStateEnergies[i] = getStateVectorEnergy1M(trialState[i].initialState, model, normAmp=fs2cm)

    end
    
    minIndex = findmin(trialStateEnergies)[2]
    
    println("Lowest energy 1M state was found: E=$(trialStateEnergies[minIndex])")
    println("LEES optimization time: $(trialOptimTime[minIndex])")

    return trialState[minIndex] 
end


function getThemalFluctuationStateVector1M(state::stateStruct, E0::Float64, dE::Float64, model::modelStruct)    
    
    @unpack pr, id, nd = model
    @unpack temp = pr

    E_target = E0 + dE
    nmtp = model.nd.nmultp1M
    
    batch_size = 200
    tol = 1.0

    count = 0
    while count < 10000

        X = map( _ -> begin
            x0 = deepcopy(state.initialState)

            @tullio  ildx[i,n,h] := getModeDisplacement(pr.iwE[n,h], temp, model) (i in 1:nmtp)
            x0[id.li1M] .+= ildx

            return x0
        end, 1:batch_size)

        E_random = map( i -> getStateVectorEnergy1M(X[i], model), 1:batch_size )

        nmatch = findfirst((abs.(E_target .- E_random) .* fs2cm) .< tol)

        if !isnothing(nmatch)
            return X[nmatch]
        end

        count += 1  
    end

    println("WARNING: Could not find thermal state with target energy:")

    for n in 1:nd.nbath1M, h in 1:nd.nmode1M
        state.initialState[id.li1M[1,n,h]] += getModeDisplacement(pr.iwE[n,h], temp, model)
    end
    return state.initialState

end

function getAbsResp(model::modelStruct, k::Int = 1; args::Dict{String, Any})
    tmodel = getTrajectoryModel(k, model)
    return getAbsRespViaField(tmodel)
end

function getAbsRespViaField(tmodel::modelStruct)

    @unpack pr, traj = tmodel
    @unpack u0, eVec = traj

    R = createState(0, u0, 0.0, 1)
    if tmodel.nd.nmultp0M > 1
        R.propagateSimply = 3
    end
    propagate!(R, pr.tmax, tmodel)
    L = createState(0, u0, 0.0, 1)
    
    k1 = transitionDipoleOperator(eVec, +1)
    L = getNewState(k1, L, tmodel)
    propagate!(L, pr.tmax, tmodel)

    kI = transitionDipoleOperator(eVec, -1, "L")

    auxData = Dict()

    return (map(t -> expValueAt(t, L, kI, R, tmodel), tmodel.pr.time), auxData)
end

function processDiagram(linResp::Vector{ComplexF64}, model::modelStruct; conv::Real = -1.0, norm::Int=0, pad::Int=0)

    @unpack pr = model

    if conv != -1.0 && conv != 0.0 && conv != 0
        linRespInside = linResp .* map(exp, -pr.time/conv)
    else
        linRespInside = deepcopy(linResp)
    end

    if pad != 0.0 && isreal(pad) == true
        newLength = nextpow(2, pad*length(linRespInside))
        newLinResp = zeros(typeof(linRespInside[1]), newLength)
        newLinResp[1:length(linRespInside)] = linRespInside
        linRespInside = newLinResp
    end

    linRespW = real(FFTW.fftshift(FFTW.fft(linRespInside)))
    freq = collect(2.0*pi*fs2cm*FFTW.fftshift(FFTW.fftfreq(length(linRespInside), 1/pr.dt)))

    Bool(norm) ? linRespW ./= maximum(linRespW) : nothing 
    
    return freq, linRespW
end

function ensembleDiagram2D(F, diagram::String, type::String, model::modelStruct; parallel::String = "t", verbose::Int = 1)

    if parallel == "n"

        return mapreduce(k -> begin
            tcalc = @elapsed x = F(diagram, type, model, parallel, k)
            verbose == 1 ? println("w:",Distributed.myid()-1," ","t:",Threads.threadid()," ", ": Trajectory ", k, " is done (", round(tcalc, digits=2), " s).") : nothing
            return x
        end, +, 1:model.nd.ntraj) ./ model.nd.ntraj

    elseif parallel == "t"

        trajDiagram = Array{Matrix{ComplexF64}}(undef, model.nd.ntraj)
        Threads.@threads for k in 1:model.nd.ntraj
            tcalc = @elapsed trajDiagram[k] = F(diagram, type, model, parallel, k)          
            verbose == 1 ? println("w:",Distributed.myid()-1," ","t:",Threads.threadid()," ", ": Trajectory ", k, " is done (", round(tcalc, digits=2), " s).") : nothing
        end
        return sum(trajDiagram) ./ model.nd.ntraj;

    elseif parallel == "pt"

        nthreads = Threads.nthreads()
        nworkers = Distributed.nprocs() - 1

        ntraj = model.nd.ntraj
        ntraj_distr = collect(Iterators.partition(1:ntraj, ceil.(Int, ntraj/nworkers)))

        ntraj_threads = collect.(Iterators.partition.(ntraj_distr, nthreads)) 
        ntraj_threads = reduce(vcat, ntraj_threads)

        n_pmap = length(ntraj_threads)

        trajDiagram = pmap(k -> begin
        
            xlen = length(ntraj_threads[k])
            x = Array{Matrix{ComplexF64}}(undef, xlen)
            Threads.@threads for i in 1:length(ntraj_threads[k])
                tcalc = @elapsed x[i] = F(diagram, type, model, parallel, ntraj_threads[k][i])            
                verbose == 1 ? println("w:",Distributed.myid()-1," ","t:",Threads.threadid()," ", ": Trajectory ", ntraj_threads[k][i], " is done (", round(tcalc, digits=2), " s).") : nothing
            end
            return sum(x);

        end, 1:n_pmap);

        return sum(trajDiagram) ./ ntraj;

    else
        error("Package Distributed.jl is required.")
    end

end

function getDiagram2D(diagram::String, type::String, model::modelStruct, distr::String, k::Int = 1)

    tmodel = getTrajectoryModel(k, model)
    
    @unpack twaiting2D, time2D = tmodel.pr
    @unpack u0, eVec = tmodel.traj
    
    if diagram == "GSB"
        if type == "R"
            F = distr == "n" ? getGSB_R_threaded : getGSB_R
        elseif type == "NR"
            F = distr == "n" ? getGSB_NR_threaded : getGSB_NR
        else
            error("Unknown diagram type.")
        end    
    elseif diagram == "ESE"
        if type == "R"
            F = distr == "n" ? getESE_R_threaded : getESE_R
        elseif type == "NR"
            F = distr == "n" ? getESE_NR_threaded : getESE_NR
        else
            error("Unknown diagram type.")
        end    
    elseif diagram == "ESA"
        if type == "R"
            F = distr == "n" ? getESA_R_threaded : getESA_R
        elseif type == "NR"
            F = distr == "n" ? getESA_NR_threaded : getESA_NR
        else
            error("Unknown diagram type.")
        end  
        
    elseif diagram == "GSB-tW-therm"
        if type == "R"
            F = distr == "n" ? error("Not implemented") : getGSB_tW_therm_R
        elseif type == "NR"
            F = distr == "n" ? error("Not implemented") : getGSB_tW_therm_NR
        else
            error("Unknown diagram type.")
        end 
    elseif diagram == "ESE-tW-therm"
        if type == "R"
            F = distr == "n" ? error("Not implemented") : getESE_tW_therm_R
        elseif type == "NR"
            F = distr == "n" ? error("Not implemented") : getESE_tW_therm_NR
        else
            error("Unknown diagram type.")
        end    
    elseif diagram == "ESA-tW-therm"
        if type == "R"
            F = distr == "n" ? error("Not implemented") : getESA_tW_therm_R
        elseif type == "NR"
            F = distr == "n" ? error("Not implemented") : getESA_tW_therm_NR
        else
            error("Unknown diagram type.")
        end    
        
    else
        error("Unknown diagram name.")
    end

    return F(u0, eVec, time2D, twaiting2D, tmodel)
end


function getEnsembleKineticEnergy(stateEnsemble::Vector{stateStruct},tmin::Real,tmax::Real,dt::Real,tau::Real,model::modelStruct)
    
    tmin = max(tmin, stateEnsemble[1].tinitial)
    tmax = min(tmax, stateEnsemble[1].tfinal)
    
    if tmax+tau > stateEnsemble[1].tfinal 
        tmax -= tau
    end
    
    trange = collect(tmin:dt:tmax)
    ntime = length(trange)
    nstate = length(stateEnsemble)

    if stateEnsemble[1].manifold == 0
        nbath = model.nd.nbath0M
        nmode = model.nd.nmode0M
        ai = model.id.ai0M
        li = model.id.li0M
        w = reshape(myModel.pr.iwG[:], 1, length(myModel.pr.iwG[:]))
    elseif stateEnsemble[1].manifold == 1
        nbath = model.nd.nbath1M
        nmode = model.nd.nmode1M
        ai = model.id.ai1M
        li = model.id.li1M
        w = model.pr.iwE
    elseif stateEnsemble[1].manifold == 2
        nbath = model.nd.nbath2M
        nmode = model.nd.nmode2M
        ai = model.id.ai2M
        li = model.id.li2M
        w = model.pr.iwE
    end

    local im2fun(state,t,s) = begin
        ca = state.dyn(t+s)[ai]
        cl = state.dyn(t+s)[li]
        S = getS(cl,cl)

        @tullio curr_norm := real(conj(ca[i,n])*ca[j,n]*S[i,j])

        @tullio x[k,q] := real <| -conj(ca[i,n])*ca[j,n]*S[i,j]*w[k,q]*( (conj(cl[i,k,q]) - cl[j,k,q])^2 ) / 4.0 / curr_norm
        
        return x
    end
        
    periodCondInd = w.*tau .< 1
    if length(w[periodCondInd]) != 0
        wCond = maximum(w[periodCondInd]).*fs2cm
        nCond = length(w[periodCondInd])
        println("Bad period condition for w < ($wCond). N: ($nCond)")
    else
        println("Period condition sufficient for all modes.")
    end

    K = [zeros(Float64, nbath, nmode) for _ in 1:ntime]

    Threads.@threads for ti in 1:ntime

        local t = trange[ti]

        local ifun(s) = begin
            im2array = zeros(Float64, nbath, nmode)
            for i in 1:nstate
                im2array .+= im2fun(stateEnsemble[i],t,s)
            end
            return im2array./nstate
        end
        
        K[ti] .+= quadgk(ifun, 0, tau)[1]
    end
    trange .+= dt/2
    return trange, K./tau
end

function getTemperatureFromStateEnsemble(stateEnsemble::Vector{stateStruct},tmin::Real,tmax::Real,dt::Real,tau::Real,model::modelStruct)
    trange, KE = getEnsembleKineticEnergy(stateEnsemble, tmin, tmax, dt, tau, model)
    stateEnsemble[1].manifold == 0 ? w = reshape(myModel.pr.iwG[:], 1, length(myModel.pr.iwG[:])) : w = model.pr.iwE
    @tullio T[t,k,q] := w[k,q]./log.(1 .+ w[k,q]./(2*KE[t][k,q]))./kbfs
    return trange, T
end
