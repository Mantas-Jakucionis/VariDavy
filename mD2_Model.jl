function createModel()
     nd = dimensionStruct()
     id = indexStruct()
     pr = parameterStruct()
     ens = ensembleStruct()
     return modelStruct(nd, id, pr, ens)
end

function getTrajectoryModel(k::Int, model::modelStruct; recomp::Int=0)

     @assert !isnothing(model.ens) && isnothing(model.traj) "Can not create trajectory model."     
    
     traj = trajectoryStruct(
          model.ens.u0[k],
          model.ens.eVec[k],
          model.ens.dE[k],
          model.ens.therm[k],
          model.ens.thermTrf[k],
          stateStruct(-1, [ComplexF64(0.0, 0.0)], 0.0, -1)
     )

     tmodel = modelStruct(model, traj)

     @tullio tmodel.pr.H[n,n] += tmodel.traj.dE[n]
     
     F = Bool(model.pr.singleMultipleLES) ? getLowestEnergyState1M_MTP1 : getLowestEnergyState1M
     recomp == 1 ? tmodel.traj.lowestEnergyState1M = F(tmodel) :
                    tmodel.traj.lowestEnergyState1M = model.ens.lowestEnergyState1M

     return tmodel
end

function createState(manifold::Int, initialState::Vector{ComplexF64}, tinitial::Float64, propagateSimply::Int)
     return deepcopy(stateStruct(manifold, initialState, tinitial, propagateSimply))
end

function createStateAt(t::Float64, state::stateStruct, model::modelStruct)
     @assert state.tinitial <= t <= state.tfinal "Can not create new state at time outside of initial state time range."
     if state.tfinal == state.tinitial
          newState = deepcopy(stateStruct(state.tinitial, state))
          dynamics = state.initialState
     else
          newState = deepcopy(stateStruct(t, state))
          dynamics = state.dyn(t)
     end
     newState.norm = getStateVectorNorm(dynamics, state.manifold, model)
     return newState
end

function initializeModel!(model::modelStruct, input)
     initializeDimensionsAndParameters!(input, model)
     initializeIndexStruct!(model)
     initializeEnsemble!(model)
     return nothing
end

function initializeEnsemble!(model::modelStruct)
     initializeEnsembleInitialConditions!(model)
     initializeEnsembleStaticEnergyShifts!(model)
     initializeEnsembleExternalFieldPolarizations!(model)
     initializeEnsembleThermalizationEvents!(model)
     return nothing
end

function initializeDimensionsAndParameters!(input, model::modelStruct)
     @unpack nd, pr = model

     pr.atol = input.atol
     pr.rtol = input.rtol
     pr.iterRtol = input.iterRtol
     pr.apopThres = input.apopThres
     pr.d2Thres = input.d2Thres
     pr.singleMultipleLES = input.singleMultipleLES
     nd.ntrialsLES = input.ntrialsLES

     pr.thermalization = input.thermalization
     pr.thermTempOffset = input.thermTempOffset
     pr.thermMethod = input.thermMethod
     pr.thermType = input.thermType
     pr.thermStep = input.thermStep
     pr.thermRate = input.thermRate
     pr.thermProb = pr.thermStep * pr.thermRate

     pr.trfThermTime = input.trfThermTime

     input.nmultp == 1 ? pr.propagateSimply = input.propagateSimply : pr.propagateSimply = 0
     pr.propagationMethod = input.propagationMethod

     pr.multpDistr = input.multpDistr
     pr.multpPerLayer = input.multpPerLayer
     pr.multpDistance = input.multpDistance

     nd.nmultp0M = input.nmultp
     nd.nmultp1M = input.nmultp
     nd.nmultp2M = input.nmultp

     nd.nsite0M = 1
     nd.nsite1M = input.nsite

     nd.nbath0M = 1
     nd.nbath1M = input.nibath
     nd.nbath2M = input.nibath

     pr.H = input.H[1:nd.nsite1M, 1:nd.nsite1M] .* cm2fs
     pr.K = 0.5 * input.K[1:nd.nsite1M, 1:nd.nsite1M] .* cm2fs
     pr.rsite = input.rsite[1:nd.nsite1M] .* cm2fs
     pr.dipGE = input.dipGE[1:nd.nsite1M, :]
     pr.dipGE ./= maximum(abs.(pr.dipGE))

     any(isnothing.([input.iwG, input.iwE, input.id])) ? begin
          pr.iwG = Float64[]
          pr.iwE = Float64[]
          pr.id = Float64[]
     end : begin
          pr.iwG = input.iwG[1:nd.nsite1M, :] .* cm2fs
          pr.iwE = input.iwE[1:nd.nsite1M, :] .* cm2fs
          pr.id = input.id[1:nd.nsite1M, :]
     end

     pr.bathSpectralDensity = input.bathSpectralDensity
     pr.bathSpdParameters = input.bathSpdParameters

     pr.staticDisorderSigma = input.staticDisorderSigma

     pr.extFieldDistribution = input.extFieldDistribution
     pr.extEstatic = input.extEstatic

     pr.tmin = input.tmin
     pr.tmax = input.tmax
     pr.dt = input.dt

     pr.temp = input.temp
     nd.ntraj = input.ntraj

     pr.twaiting2D = input.twaiting2D
     pr.tmin2D = input.tmin2D
     pr.tmax2D = input.tmax2D
     pr.dt2D = input.dt2D


     disParameter = [pr.staticDisorderSigma]
     pr.disordered = all(Bool.(disParameter .== 0.0)) ? 0 : 1

     if nd.nsite1M > 1
          pr.particleType = input.particleType
     elseif nd.nsite1M == 1 && input.particleType == "fermion"
          pr.particleType = "boson"
          println("Switching to boson particple type.")
     else
          pr.particleType = "boson"
     end

     # Computing additional variables from input
     if pr.particleType == "boson"
          nd.nsite2M = Int(nd.nsite1M * (nd.nsite1M + 1) / 2)
     elseif pr.particleType == "fermion"
          nd.nsite2M = Int(length(collect(combinations(1:nd.nsite1M, 2))))
     else
          error("Unknown particle type.")
     end

     pr.time = collect(pr.tmin:pr.dt:pr.tmax)
     nd.ntime = length(pr.time)
     pr.time2D = collect(pr.tmin2D:pr.dt2D:pr.tmax2D)

     # Appending
     pr.wmin = input.wmin
     pr.wmax = input.wmax
     pr.dw = input.dw

     #
     wbase = collect(input.wmin:input.dw:input.wmax) .* cm2fs
     nmode = length(wbase)

     pr.dwp = input.dwp

     if nmode > 0
          wG = zeros(Float64, nd.nsite1M, nmode)
          wE = zeros(Float64, nd.nsite1M, nmode)
          for k in 1:nd.nsite1M
               wG[k, :] = wbase
               wE[k, :] = wbase .* pr.dwp
          end
          g = zeros(Float64, nd.nsite1M, nmode)
          if input.wmax > input.wmin
               @tullio J[n, q] := pr.bathSpectralDensity(wE[n, q], pr.bathSpdParameters)
               @tullio g[n, q] = input.dw * sqrt(J[n, q] / pi) / wE[n, q]
               @tullio rtemp[n] := g[n, q]^2 * wE[n, q]
               @tullio g[n, q] *= sqrt(pr.rsite[n] / rtemp[n])
          end

          all(length.([pr.iwG, pr.iwE, pr.id]) .== 0) ? begin
               pr.iwG = wG
               pr.iwE = wE
               pr.id = g .* sqrt(2.0)
          end : begin
               pr.iwG = hcat(pr.iwG, wG)
               pr.iwE = hcat(pr.iwE, wE)
               pr.id = hcat(pr.id, g .* sqrt(2.0))
          end
     end

     pr.diw = pr.iwE - pr.iwG
     pr.anharmQ = any(pr.diw .!= 0.0)

     pr.f = pr.id ./ sqrt(2.0)

     nd.nmode0M = Int(length(pr.iwG) ./ nd.nsite0M)
     nd.nmode1M = Int(length(pr.iwE) ./ nd.nsite1M)
     nd.nmode2M = nd.nmode1M

     nd.astatesize0M = nd.nmultp0M * nd.nsite0M
     nd.lstatesize0M = nd.nmultp0M * nd.nbath0M * nd.nmode0M
     nd.statesize0M = nd.astatesize0M + nd.lstatesize0M

     nd.astatesize1M = nd.nmultp1M * nd.nsite1M
     nd.lstatesize1M = nd.nmultp1M * nd.nbath1M * nd.nmode1M
     nd.statesize1M = nd.astatesize1M + nd.lstatesize1M

     nd.astatesize2M = nd.nmultp2M * nd.nsite2M
     nd.lstatesize2M = nd.nmultp2M * nd.nbath2M * nd.nmode2M
     nd.statesize2M = nd.astatesize2M + nd.lstatesize2M

     return nothing
end


function initializeIndexStruct!(model::modelStruct)
     @unpack nd, id, pr = model

    id.Dmt   = ones(ComplexF64, nd.nmultp1M)
    id.Dn    = ones(ComplexF64, nd.nsite1M)
    id.Dmt0M = ones(ComplexF64, nd.nmultp0M)
    id.Dn0M  = ones(ComplexF64, nd.nsite0M)
    id.Dn2M  = ones(ComplexF64, nd.nsite2M)

    id.ai0M    = Array{Int}(undef, nd.nmultp0M, nd.nsite0M)
    id.li0M   = Array{Int}(undef, nd.nmultp0M, nd.nbath0M, nd.nmode0M)

    id.ai1M    = Array{Int}(undef, nd.nmultp1M, nd.nsite1M)
    id.li1M   = Array{Int}(undef, nd.nmultp1M, nd.nbath1M, nd.nmode1M)

    id.ai1M_MTP1    = Array{Int}(undef, nd.nsite1M)
    id.li1M_MTP1   = Array{Int}(undef, nd.nbath1M, nd.nmode1M)

    id.ai2M    = Array{Int}(undef, nd.nmultp1M, nd.nsite2M)
    id.li2M   = Array{Int}(undef, nd.nmultp1M, nd.nbath2M, nd.nmode2M)

    for mt in 1:nd.nmultp0M
          for n in 1:nd.nsite0M
               id.ai0M[mt,n] = (n-1)*nd.nmultp0M + (mt-1) + 1
          end
     end

     for mt in 1:nd.nmultp0M
          for k in 1:nd.nbath0M
               for q in 1:nd.nmode0M
                    id.li0M[mt,k,q] = ( nd.nmultp0M*( (q-1)*nd.nbath0M + (k-1) ) + (mt-1) ) + 1 + nd.astatesize0M
               end
          end
     end

    for mt in 1:nd.nmultp1M
          for n in 1:nd.nsite1M
               id.ai1M[mt,n] = (n-1)*nd.nmultp1M + (mt-1) + 1
          end
     end
     
     for mt in 1:nd.nmultp1M
          for n in 1:nd.nbath1M
               for h in 1:nd.nmode1M
                    id.li1M[mt,n,h] = ( nd.nmultp1M*( (h-1)*nd.nbath1M + (n-1) ) + (mt-1) ) + 1 + nd.astatesize1M
               end
          end
     end
     
     for n in 1:nd.nsite1M
          id.ai1M_MTP1[n] = (n-1)*1 + 1
     end
     
     for n in 1:nd.nbath1M
          for h in 1:nd.nmode1M
               id.li1M_MTP1[n,h] = ( 1*( (h-1)*nd.nbath1M + (n-1) ) ) + 1 + length(id.ai1M_MTP1)
          end
     end


     for mt in 1:nd.nmultp2M
          for nm in 1:nd.nsite2M
               id.ai2M[mt,nm] = (nm-1)*nd.nmultp1M + (mt-1) + 1
          end
     end

     for mt in 1:nd.nmultp2M
          for n in 1:nd.nbath2M
               for h in 1:nd.nmode2M
                    id.li2M[mt,n,h] = ( nd.nmultp1M*( (h-1)*nd.nbath2M + (n-1) ) + (mt-1) ) + 1 + nd.astatesize2M
               end
          end
     end

    id.aRi0M   = Array{Int}(undef, nd.nmultp0M, nd.nsite0M)
    id.aIi0M   = Array{Int}(undef, nd.nmultp0M, nd.nsite0M)
    id.lRi0M  = Array{Int}(undef, nd.nmultp0M, nd.nbath0M, nd.nmode0M)
    id.lIi0M  = Array{Int}(undef, nd.nmultp0M, nd.nbath0M, nd.nmode0M)

    for mt in 1:nd.nmultp0M
         for n in 1:nd.nsite0M
              id.aRi0M[mt,n] = (n-1)*2*nd.nmultp0M + 2*(mt-1) + 1               
              id.aIi0M[mt,n] = (n-1)*2*nd.nmultp0M + 2*(mt-1) + 2           
         end
    end

    for mt in 1:nd.nmultp0M
         for k in 1:nd.nbath0M
              for q in 1:nd.nmode0M
                   id.lRi0M[mt,k,q] = (2*nd.astatesize0M) + 2*( nd.nmultp0M*( (q-1)*nd.nbath0M + (k-1) ) + (mt-1) ) + 1
                   id.lIi0M[mt,k,q] = (2*nd.astatesize0M) + 2*( nd.nmultp0M*( (q-1)*nd.nbath0M + (k-1) ) + (mt-1) ) + 2
              end
         end
    end

    id.aRi1M   = Array{Int}(undef, nd.nmultp1M, nd.nsite1M)
    id.aIi1M   = Array{Int}(undef, nd.nmultp1M, nd.nsite1M)
    id.lRi1M  = Array{Int}(undef, nd.nmultp1M, nd.nbath1M, nd.nmode1M)
    id.lIi1M  = Array{Int}(undef, nd.nmultp1M, nd.nbath1M, nd.nmode1M)

    for mt in 1:nd.nmultp1M
         for n in 1:nd.nsite1M
              id.aRi1M[mt,n] = (n-1)*2*nd.nmultp1M + 2*(mt-1) + 1                 
              id.aIi1M[mt,n] = (n-1)*2*nd.nmultp1M + 2*(mt-1) + 2                 
         end
    end

    for mt in 1:nd.nmultp1M
         for n in 1:nd.nbath1M
              for h in 1:nd.nmode1M
                   id.lRi1M[mt,n,h] = (2*nd.astatesize1M) + 2*( nd.nmultp1M*( (h-1)*nd.nbath1M + (n-1) ) + (mt-1) ) + 1
                   id.lIi1M[mt,n,h] = (2*nd.astatesize1M) + 2*( nd.nmultp1M*( (h-1)*nd.nbath1M + (n-1) ) + (mt-1) ) + 2 
              end
         end
    end

    id.aRi2M   = Array{Int}(undef, nd.nmultp2M, nd.nsite2M)
    id.aIi2M   = Array{Int}(undef, nd.nmultp2M, nd.nsite2M)
    id.lRi2M  = Array{Int}(undef, nd.nmultp2M, nd.nbath2M, nd.nmode2M)
    id.lIi2M  = Array{Int}(undef, nd.nmultp2M, nd.nbath2M, nd.nmode2M)

    for mt in 1:nd.nmultp2M
         for n in 1:nd.nsite2M
              id.aRi2M[mt,n] = (n-1)*2*nd.nmultp2M + 2*(mt-1) + 1                 
              id.aIi2M[mt,n] = (n-1)*2*nd.nmultp2M + 2*(mt-1) + 2                 
         end
    end

    for mt in 1:nd.nmultp2M
         for n in 1:nd.nbath2M
              for h in 1:nd.nmode2M
                   id.lRi2M[mt,n,h] = (2*nd.astatesize2M) + 2*( nd.nmultp2M*( (h-1)*nd.nbath2M + (n-1) ) + (mt-1) ) + 1
                   id.lIi2M[mt,n,h] = (2*nd.astatesize2M) + 2*( nd.nmultp2M*( (h-1)*nd.nbath2M + (n-1) ) + (mt-1) ) + 2 
              end
         end
    end

    id.i12 = Array{Int, 2}(undef, nd.nsite1M, nd.nsite1M)
    id.i21 = Array{Tuple{Int, Int}}(undef, nd.nsite2M)
    fill!(id.i12, -1)
    ind = 1
    if pr.particleType == "boson"
          for m in 1:nd.nsite1M
               for n in 1:m
                    id.i12[n,m] = ind
                    id.i21[ind] = (n, m)
                    ind += 1
               end
          end
     elseif pr.particleType == "fermion"
          x = collect(combinations(1:nd.nsite1M,2))
          id.i21 = [Tuple(x[i]) for i in 1:length(x)]
          for i in 1:length(x)
               id.i12[x[i][1], x[i][2]] = ind
               ind += 1
          end
     end

    id.N1 = zeros(Float64, nd.nsite1M, nd.nsite2M, nd.nsite1M, nd.nsite2M)
    id.N2 = zeros(Float64, nd.nsite1M, nd.nsite1M, nd.nsite2M, nd.nsite2M)

    for kl in 1:nd.nsite2M
        for nm in 1:nd.nsite2M
               k, l = id.i21[kl]
               n, m = id.i21[nm]
               if pr.particleType == "boson"
                    C = sqrt(2.0)
                    for v in 1:nd.nsite1M
                         for u in 1:nd.nsite1M
                              id.N1[v,kl,u,nm] += (δ(k,l)*(δ(v,k)*C-δ(v,l))+δ(v,l))*(δ(n,m)*(δ(u,n)*C-δ(u,m))+δ(u,m))*δ(k,n)
                              id.N1[v,kl,u,nm] += (δ(k,l)*(δ(v,k)*C-δ(v,l))+δ(v,l))*(1-δ(n,m))*δ(u,n)*δ(k,m)
                              id.N1[v,kl,u,nm] += (δ(n,m)*(δ(u,n)*C-δ(u,m))+δ(u,m))*(1-δ(k,l))*δ(v,k)*δ(l,n)
                              id.N1[v,kl,u,nm] += (1-δ(n,m))*(1-δ(k,l))*δ(v,k)*δ(u,n)*δ(l,m)

                              id.N2[v,u,kl,nm] += (δ(k,l)*(δ(v,k)*C-δ(v,l))+δ(v,l))*(δ(n,m)*(δ(u,n)*C-δ(u,m))+δ(u,m))*δ(v,n)*δ(u,k)
                              id.N2[v,u,kl,nm] += (δ(n,m)*(δ(u,n)*C-δ(u,m))+δ(u,m))*(1-δ(k,l))*δ(u,l)*δ(v,n)*δ(v,k)
                              id.N2[v,u,kl,nm] += (δ(k,l)*(δ(v,k)*C-δ(v,l))+δ(v,l))*(1-δ(n,m))*δ(v,m)*δ(u,k)*δ(u,n)
                              id.N2[v,u,kl,nm] += (1-δ(n,m))*(1-δ(k,l))*δ(u,n)*δ(v,m)*δ(v,k)*δ(u,l)

                         end
                    end
               elseif pr.particleType == "fermion"
                    for v in 1:nd.nsite1M
                         for u in 1:nd.nsite1M
                              id.N1[v,kl,u,nm] += δ(v,l)*(δ(u,m)*δ(k,n) + δ(u,n)*δ(k,m))
                              id.N1[v,kl,u,nm] += δ(v,k)*(δ(u,m)*δ(l,n) + δ(u,n)*δ(l,m))

                              id.N2[v,u,kl,nm] += δ(u,m)*δ(v,n)*(δ(v,l)*δ(u,k) + δ(u,l)*δ(v,k))
                              id.N2[v,u,kl,nm] += δ(u,n)*δ(v,m)*(δ(v,l)*δ(u,k) + δ(v,k)*δ(u,l))
                         end
                    end
               end
          end
     end

     return nothing
end
