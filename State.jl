
function getStateVectorNorm(sVec::Array{ComplexF64}, m::Int, model::modelStruct)
     ca = getAlpha(sVec, m, model)
     cl = getLambda(sVec, m, model)
     S = getS(cl, cl)
     return real(@tullio norm := conj(ca[i, n]) * ca[j, n] * S[i, j])
end

function getNormalizedStateVector0M(sVec::Array{ComplexF64}, model::modelStruct, fN::Float64=1.0)
     @unpack id = model
     norm = getStateVectorNorm(sVec, 0, model)
     ca = getAlpha(sVec, 0, model)
     sVecCopy = deepcopy(sVec)
     sVecCopy[id.ai0M] .= ca[id.ai0M] .* fN ./ sqrt(norm)
     return sVecCopy
end

function getNormalizedStateVector1M(sVec::Array{ComplexF64}, model::modelStruct, fN::Float64=1.0)
     @unpack id = model
     norm = getStateVectorNorm(sVec, 1, model)
     ca = getAlpha(sVec, 1, model)

     sVecCopy = deepcopy(sVec)
     sVecCopy[id.ai1M] .= ca[id.ai1M] .* fN ./ sqrt(norm)
     return sVecCopy
end

function getNormalizedStateVector2M(sVec::Array{ComplexF64}, model::modelStruct, fN::Float64=1.0)
     @unpack id = model
     norm = getStateVectorNorm(sVec, 2, model)
     ca = getAlpha(sVec, 2, model)
     sVecCopy = deepcopy(sVec)
     sVecCopy[id.ai2M] .= ca[id.ai2M] .* fN ./ sqrt(norm)
     return sVecCopy
end

function getStateVectorFirstMultipleProjection(sVec::Vector{ComplexF64}, manifold::Int, model::modelStruct)

     if manifold == 0
          ai = model.id.ai0M
          li = model.id.li0M
     elseif manifold == 1
          ai = model.id.ai1M
          li = model.id.li1M
     elseif manifold == 2
          ai = model.id.ai2M
          li = model.id.li2M
     else
          error("Unknown manifold.")
     end
          
     ca = getAlpha(sVec, manifold, model)
     cl = getLambda(sVec, manifold, model)
     S  = getS(cl, cl)
 
     @tullio prob[n] := real <| conj(ca[i,n])*ca[j,n]*S[i,j]
     norm = sum(prob)
 
     @tullio x[k,q] := real <|    conj(ca[i,n])*ca[j,n]*S[i,j]*(conj(cl[i,k,q]) + cl[j,k,q]) / sqrt(2.0) / norm
     @tullio p[k,q] := real <| im*conj(ca[i,n])*ca[j,n]*S[i,j]*(conj(cl[i,k,q]) - cl[j,k,q]) / sqrt(2.0) / norm
 
     out = zeros(eltype(sVec), size(sVec))

     @tullio cl0[1,k,q] := ComplexF64(x[k,q], p[k,q]) / sqrt(2.0)
     S0 = getS(cl0,cl)

     @tullio out[ai[1,n]] = ca[i,n]*S0[1,i] 
     @tullio out[li[1,k,q]] = cl0[1,k,q]

     return out
 end

function projectionSOQ(kap, par)
     
     ca, cl, cl_kap, cl_flat, norm, mu_x, mu_p, s_x, s_p = par
     
     nmultp  = size(cl,1)

     kap_cmplx = reinterpret(ComplexF64, kap)
     sz = Int(length(kap_cmplx)/(nmultp))
     kap = [kap_cmplx[(i-1)*sz + 1 : i*sz] for i in 1:nmultp]

     Cp = getXn(kap, nmultp)
     Cm = getPn(kap, nmultp)
          
     @tullio K[i,j] := exp <| conj(kap[i][p])*kap[j][p] - 0.5*abs2(kap[i][p]) - 0.5*abs2(kap[j][p])
     
     @tullio D[i,j] := exp <| conj(kap[i][p])*cl_kap[j][p] - 0.5*abs2(kap[i][p]) - 0.5*abs2(cl_kap[j][p])
          
     @tullio Sk[i,j] := conj(cl_flat[i][q])*cl_flat[j][q] - 0.5*abs2(cl_flat[i][q]) - 0.5*abs2(cl_flat[j][q])
     @tullio Sk[i,j] += -conj(cl_kap[i][p])*cl_kap[j][p]  + 0.5*abs2(cl_kap[i][p])  + 0.5*abs2(cl_kap[j][p])
     map!(exp, Sk, Sk)

     @tullio A[i] := conj(D[i,j])*Sk[j,i]
     @tullio B[i] := conj(K[i,j])*Sk[j,i]
     @tullio beta[i,n] := ca[i,n]*A[i]/B[i]

     out = zeros(Float64, 2, size(kap,1), size(kap[1],1))
     
     @tullio out[1,n,p] += real <| conj(beta[i,m])*beta[j,m]*K[i,j]*Sk[i,j]*Cp[i,j,p,n] ./ norm
     @tullio out[2,n,p] += real <| conj(beta[i,m])*beta[j,m]*K[i,j]*Sk[i,j]*Cm[i,j,p,n] ./ norm

     @tullio out[1,n,p] += -getNormalDistributionMoments(mu_x[p], s_x[p], n)
     @tullio out[2,n,p] += -getNormalDistributionMoments(mu_p[p], s_p[p], n)

     return sum(abs.(out))
end

function getNormalDistributionMoments(mu, s, n)

     if n == 1
          return mu
     elseif n == 2
          return mu^2 + s
     elseif n == 3
          return mu^3 + 3 * mu * s^2
     elseif n == 4
          return mu^4 + 6 * mu^2 * s^2 + 3 * s^4
     elseif n == 5
          return mu^5 + 10 * mu^3 * s^2 + 15 * mu * s^4
     else
          error("Non-central moment of $(n) order is not yet implemented.")
     end

end

function getXn(kap, N::Int)

     nmultp = size(kap, 1)
     nmode = size(kap[1], 1)

     C = zeros(ComplexF64, nmultp, nmultp, nmode, N)

     for n in 1:N
          for k in 0:n 
               for l in 0:(n-k) 
                    for i in 1:nmultp, j in 1:nmultp, p in 1:nmode
                         C[i,j,p,n] += getAcoef(n,k) * binomial(n-k,l) * conj(kap[i][p])^l * kap[j][p]^(n-k-l)
                    end
               end
          end
          C[:,:,:,n] ./= 2^(-n/2)
     end

     return real.(C)
end

function getPn(kap, N::Int)

     nmultp = size(kap, 1)
     nmode = size(kap[1], 1)

     C = zeros(ComplexF64, nmultp, nmultp, nmode, N)
     for n in 1:N
          for k in 0:n 
               for l in 0:(n-k) 
                    for i in 1:nmultp, j in 1:nmultp, p in 1:nmode
                         C[i,j,p,n] += getAcoef(n,k) * (-1)^k * binomial(n-k,l) * conj(kap[i][p])^l * (-kap[j][p])^(n-k-l)
                    end
               end
          end
          C[:,:,:,n] ./= im^n * 2^(-n/2)
     end

     return real.(C)
end

function getAcoef(n,i)
     if i%2 != 0
          return 0.0
     else
          return factorial(n)/factorial(n-i)/doublefactorial(i)
     end
end


function thermalizeVibrationalModes!(integrator)

     # Assumes identical sets of vibrational mode frequencies coupled to each molecule.
     
     if typeof(integrator.p) == modelStruct
          model = integrator.p
     elseif typeof(integrator.p) == cachePackStructFloat64
          @unpack model = integrator.p
     end

     @unpack nd, id, pr, traj = model

     x = integrator.u
     
     times = traj.therm.times
     ind = findfirst(times .== integrator.t)
     e = @view traj.therm.events[ind, :]

     fun = string(integrator.f.f)

     targetTemp = pr.temp + pr.thermTempOffset
     
     if contains(fun, "0M")
          manifold = 0
          e = Bool.(reshape(e, nd.nbath0M, nd.nmode0M))
          li = id.li0M[1,:,:][e]
          w = pr.iwG[:]
          w = w[e[1, :]]
     elseif contains(fun, "1M")
          manifold = 1
          e = Bool.(reshape(e, nd.nbath1M, nd.nmode1M))
          li = id.li1M[1,:,:][e]
          w = pr.iwE[e]
     elseif contains(fun, "2M")
          manifold = 2
          e = Bool.(reshape(e, nd.nbath2M, nd.nmode2M))
          li = id.li2M[1,:,:][e]
          w = pr.iwE[e]
     end

     @assert manifold != -1 "Unknown state manifold during thermalization."

     if pr.thermMethod == "simple"

          nd.nmultp1M > 1 ? x .= getStateVectorFirstMultipleProjection(x, manifold, model) : nothing
          
          clR = real(getLambda(x, manifold, model))
          clR = clR[1,:,:][e]

          @tullio x[li[i]] = ComplexF64(clR[i], imag(getModeDisplacement(w[i], targetTemp, model)))
          
          nd.nmultp1M > 1 ? setUnpopulatedMultiples!(x, manifold, model) : nothing    

     elseif pr.thermMethod == "simple-project"

          x .= getStateVectorFirstMultipleProjection(x, manifold, model)
          setUnpopulatedMultiples!(x, manifold, model)    

     end

     return nothing
end


function thermalizeVibrationalModesTrf!(integrator)

     # Assumes identical sets of vibrational mode frequencies coupled to each molecule.
     
     if typeof(integrator.p) == modelStruct
          model = integrator.p
     elseif typeof(integrator.p) == cachePackStructFloat64
          @unpack model = integrator.p
     end

     @unpack nd, id, pr, traj = model

     x = integrator.u
     
     times = traj.thermTrf.times
     ind = findfirst(times .== integrator.t)
     e = @view traj.thermTrf.events[ind, :]

     fun = string(integrator.f.f)

     targetTemp = pr.temp + pr.thermTempOffset
     
     if contains(fun, "0M")
          manifold = 0
          e = Bool.(reshape(e, nd.nbath0M, nd.nmode0M))
          li = id.li0M[1,:,:][e]
          w = pr.iwG[:]
          w = w[e[1, :]]
     elseif contains(fun, "1M")
          manifold = 1
          e = Bool.(reshape(e, nd.nbath1M, nd.nmode1M))
          li = id.li1M[1,:,:][e]
          w = pr.iwE[e]
     elseif contains(fun, "2M")
          manifold = 2
          e = Bool.(reshape(e, nd.nbath2M, nd.nmode2M))
          li = id.li2M[1,:,:][e]
          w = pr.iwE[e]
     end

     @assert manifold != -1 "Unknown state manifold during thermalization."

     nd.nmultp1M > 1 ? x .= getStateVectorFirstMultipleProjection(x, manifold, model) : nothing

     clR = real(getLambda(x, manifold, model))
     clR = clR[1,:,:][e]

     @tullio x[li[i]] = ComplexF64(clR[i], imag(getModeDisplacement(w[i], targetTemp, model)))
     
     nd.nmultp1M > 1 ? setUnpopulatedMultiples!(x, manifold, model) : nothing   

     return nothing
end


function propagate!(state::stateStruct, tfinal::Real, model::modelStruct; therm::String = "default", tW_therm_t::Tuple{Float64, Float64} = (-1.0, -1.0))

     @unpack nd, pr = model

     @assert isnothing(state.dyn) "Propagating the same state twice."
     @assert tfinal >= state.tinitial "Propagation backwards in time is not supported."

     state.tfinal = tfinal

     timeInterval = (state.tinitial, state.tfinal)

     if therm == "trf"
          thermTimes = model.traj.thermTrf.times
          thermCallBack = PresetTimeCallback(thermTimes, thermalizeVibrationalModesTrf!)
     elseif therm == "tW-therm"
          thermTimes = deepcopy(model.traj.thermTrf.times)
          thermTimes[thermTimes .< tW_therm_t[1]] .= -1.0
          thermTimes[thermTimes .> tW_therm_t[2]] .= -1.0
          thermCallBack = PresetTimeCallback(thermTimes, thermalizeVibrationalModesTrf!)
     elseif therm == "default"
          thermTimes = model.traj.therm.times
          thermCallBack = PresetTimeCallback(thermTimes, thermalizeVibrationalModes!)
     else
          error("Unknown thermalization variable value.")
     end

     if state.propagateSimply == 0

          if state.manifold == 0
               M = Array{Float64}(undef, 2 * nd.statesize0M, 2 * nd.statesize0M)
               RHS = Array{ComplexF64}(undef, nd.statesize0M)
               cachePack = cachePackStructFloat64(model, M, RHS)
               prob = ODEProblem{true}(mD2_0M_ODE_RI_APO!, state.initialState, timeInterval, cachePack)
               state.dyn = solve(prob, pr.propagationMethod, abstol=pr.atol, reltol=pr.rtol, maxiters=5e5, callback=thermCallBack)
          elseif state.manifold == 1
               M = Array{Float64}(undef, 2 * nd.statesize1M, 2 * nd.statesize1M)
               RHS = Array{ComplexF64}(undef, nd.statesize1M)
               cachePack = cachePackStructFloat64(model, M, RHS)
               prob = ODEProblem{true}(mD2_1M_ODE_RI_APO!, state.initialState, timeInterval, cachePack)
               state.dyn = solve(prob, pr.propagationMethod, abstol=pr.atol, reltol=pr.rtol, maxiters=5e5, callback=thermCallBack)
          elseif state.manifold == 2
               M = Array{Float64}(undef, 2 * nd.statesize2M, 2 * nd.statesize2M)
               RHS = Array{ComplexF64}(undef, nd.statesize2M)
               cachePack = cachePackStructFloat64(model, M, RHS)
               prob = ODEProblem{true}(mD2_2M_ODE_RI_APO!, state.initialState, timeInterval, cachePack)
               state.dyn = solve(prob, pr.propagationMethod, abstol=pr.atol, reltol=pr.rtol, maxiters=5e5, callback=thermCallBack)
          end

     elseif state.propagateSimply == 1

          if state.manifold == 0

               if nd.nmultp0M == 1
                    if model.pr.propagateSimply == 1
                         # thermCallBack = PresetTimeCallback(thermTimes, thermalizeVibrationalModes!)
                         prob = ODEProblem{true}(D2_0M!, state.initialState, timeInterval, model)
                         state.dyn = solve(prob, pr.propagationMethod, abstol=pr.atol, reltol=pr.rtol, callback=thermCallBack)
                    elseif model.pr.propagateSimply == 0
                         M = Array{Float64}(undef, 2 * nd.statesize0M, 2 * nd.statesize0M)
                         RHS = Array{ComplexF64}(undef, nd.statesize0M)
                         cachePack = cachePackStructFloat64(model, M, RHS)
                         prob = ODEProblem{true}(mD2_0M_ODE_RI_APO!, state.initialState, timeInterval, cachePack)
                         state.dyn = solve(prob, pr.propagationMethod, abstol=pr.atol, reltol=pr.rtol, callback=thermCallBack)
                    end
               elseif nd.nmultp0M > 1
                    if model.pr.propagateSimply == 1
                         setUnpopulatedMultiples!(state.initialState, 0, model)
                         prob = ODEProblem{true}(D2_0M_MTP1!, state.initialState, timeInterval, model)
                         state.dyn = solve(prob, pr.propagationMethod, abstol=pr.atol, reltol=pr.rtol, callback=thermCallBack)
                    elseif model.pr.propagateSimply == 0
                         M = Array{Float64}(undef, 2 * nd.statesize0M, 2 * nd.statesize0M)
                         RHS = Array{ComplexF64}(undef, nd.statesize0M)
                         cachePack = cachePackStructFloat64(model, M, RHS)
                         prob = ODEProblem{true}(mD2_0M_ODE_RI_APO!, state.initialState, timeInterval, cachePack)
                         state.dyn = solve(prob, pr.propagationMethod, abstol=pr.atol, reltol=pr.rtol, callback=thermCallBack)
                    end
               end

          elseif state.manifold == 1
               prob = ODEProblem{true}(D2_1M!, state.initialState, timeInterval, model)
               state.dyn = solve(prob, pr.propagationMethod, abstol=pr.atol, reltol=pr.rtol, callback=thermCallBack)
          elseif state.manifold == 2
               prob = ODEProblem{true}(D2_2M!, state.initialState, timeInterval, model)
               state.dyn = solve(prob, pr.propagationMethod, abstol=pr.atol, reltol=pr.rtol, callback=thermCallBack)
          end

          # For when mixing of multiples in the 0 excitation manifold is possible.
     elseif state.propagateSimply == 2 && nd.nmultp0M > 1 && state.manifold == 0
          M = Array{Float64}(undef, 2 * nd.statesize0M, 2 * nd.statesize0M)
          RHS = Array{ComplexF64}(undef, nd.statesize0M)
          cachePack = cachePackStructFloat64(model, M, RHS)
          prob = ODEProblem{true}(mD2_0M_ODE_RI_APO!, state.initialState, timeInterval, cachePack)
          state.dyn = solve(prob, pr.propagationMethod, abstol=pr.atol, reltol=pr.rtol, callback=thermCallBack)
     elseif state.propagateSimply == 3 && nd.nmultp0M > 1 && state.manifold == 0
          setUnpopulatedMultiples!(state.initialState, 0, model)
          prob = ODEProblem{true}(D2_0M_MTP1!, state.initialState, timeInterval, model)
          state.dyn = solve(prob, pr.propagationMethod, abstol=pr.atol, reltol=pr.rtol, callback=thermCallBack)
     else
          error("Unknown propagation simplicity number.")
     end

     state.norm = getStateVectorNorm(state.dyn(state.tfinal), state.manifold, model)

     return nothing
end

function getNewState(operator::transitionDipoleOperator, state::stateStruct, model::modelStruct)
     if state.manifold == 0
          return getNewStateByOperator0M(state, operator, model)
     elseif state.manifold == 1
          return getNewStateByOperator1M(state, operator, model)
     elseif state.manifold == 2
          return getNewStateByOperator2M(state, operator, model)
     end
end

function getNewStateByOperator0MfromDipGE(state0M::stateStruct, operator::transitionDipoleOperator, dipGE::Matrix{Float64}, model::modelStruct)

     @unpack nd, id, pr = model

     @assert operator.action == +1 "Undefined operator action."

     state1M = createState(1, Array{ComplexF64}(undef, nd.statesize1M), state0M.tfinal, pr.propagateSimply)

     mE = dipGE * operator.polarizationVector

     dynamics = (state0M.tfinal == state0M.tinitial) ? state0M.initialState : state0M.dyn(state1M.tinitial)

     for mt in 1:nd.nmultp1M
          for n in 1:nd.nsite1M
               state1M.initialState[id.ai1M[mt, n]] = dynamics[id.ai0M[mt, 1]] * mE[n]
          end
          state1M.initialState[id.li1M[mt, :, :]] = dynamics[id.li0M[mt, 1, :]]
     end

     state1M.norm = getStateVectorNorm(state1M.initialState, 1, model)

     return state1M
end


function getNewStateByOperator0M(state0M::stateStruct, operator::transitionDipoleOperator, model::modelStruct)

     @unpack nd, id, pr = model

     @assert operator.action == +1 "Undefined operator action."

     state1M = createState(1, Array{ComplexF64}(undef, nd.statesize1M), state0M.tfinal, pr.propagateSimply)

     mE = pr.dipGE * operator.polarizationVector

     dynamics = (state0M.tfinal == state0M.tinitial) ? state0M.initialState : state0M.dyn(state1M.tinitial)

     for mt in 1:nd.nmultp1M
          for n in 1:nd.nsite1M
               state1M.initialState[id.ai1M[mt, n]] = dynamics[id.ai0M[mt, 1]] * mE[n]
          end
          state1M.initialState[id.li1M[mt, :, :]] = dynamics[id.li0M[mt, 1, :]]
     end

     state1M.norm = getStateVectorNorm(state1M.initialState, 1, model)

     return state1M
end


function getNewStateByOperator1M(state1M::stateStruct, operator::transitionDipoleOperator, model::modelStruct)

     @unpack nd, id, pr = model

     mE = pr.dipGE * operator.polarizationVector

     if operator.action == -1

          state0M = createState(0, zeros(ComplexF64, nd.statesize0M), state1M.tfinal, pr.propagateSimply)

          dynamics = (state1M.tfinal == state1M.tinitial) ? state1M.initialState : state1M.dyn(state0M.tinitial)

          for mt in 1:nd.nmultp0M
               for n in 1:nd.nsite1M
                    state0M.initialState[id.ai0M[mt, 1]] += dynamics[id.ai1M[mt, n]] * mE[n]
               end
               state0M.initialState[id.li0M[mt, 1, :]] = dynamics[id.li1M[mt, :, :]]
          end

          state0M.norm = getStateVectorNorm(state0M.initialState, 0, model)

          return state0M

     elseif operator.action == +1

          state2M = createState(2, Array{ComplexF64}(undef, nd.statesize2M), state1M.tfinal, pr.propagateSimply)

          dynamics = (state1M.tfinal == state1M.tinitial) ? state1M.initialState : state1M.dyn(state2M.tinitial)

          if pr.particleType == "boson"

               for mt in 1:nd.nmultp2M
                    for n in 1:nd.nsite1M
                         nn = id.i12[n, n]
                         state2M.initialState[id.ai2M[mt, nn]] = sqrt(2.0) * dynamics[id.ai1M[mt, n]] * mE[n]
                         for m in 1:nd.nsite1M
                              if n < m
                                   nm = id.i12[n, m]
                                   state2M.initialState[id.ai2M[mt, nm]] = (dynamics[id.ai1M[mt, n]] * mE[m] + dynamics[id.ai1M[mt, m]] * mE[n])
                              end
                         end
                    end
               end

          elseif pr.particleType == "fermion"

               for mt in 1:nd.nmultp2M
                    for n in 1:nd.nsite1M
                         for m in 1:nd.nsite1M
                              if n < m
                                   nm = id.i12[n, m]
                                   state2M.initialState[id.ai2M[mt, nm]] = (dynamics[id.ai1M[mt, n]] * mE[m] + dynamics[id.ai1M[mt, m]] * mE[n])
                              end
                         end
                    end
               end

          end

          for mt in 1:nd.nmultp2M
               state2M.initialState[id.li2M[mt, :, :]] = dynamics[id.li1M[mt, :, :]]
          end

          state2M.norm = getStateVectorNorm(state2M.initialState, 2, model)

          return state2M
     else
          error("Undefined operator action.")
     end

end

function getNewStateByOperator2M(state2M::stateStruct, operator::transitionDipoleOperator, model::modelStruct)

     @unpack nd, id, pr = model

     if state2M.manifold == 2

          @assert operator.action == -1 "Undefined operator action."

          state1M = createState(1, zeros(ComplexF64, nd.statesize1M), state2M.tfinal, pr.propagateSimply)

          dynamics = (state2M.tfinal == state2M.tinitial) ? state2M.initialState : state2M.dyn(state1M.tinitial)

          mE = pr.dipGE * operator.polarizationVector

          if pr.particleType == "boson"

               for mt in 1:nd.nmultp1M
                    for n in 1:nd.nsite1M
                         nn = id.i12[n, n]
                         state1M.initialState[id.ai1M[mt, n]] += sqrt(2.0) * dynamics[id.ai2M[mt, nn]] * mE[n]
                         for m in 1:nd.nsite1M
                              if m < n
                                   mn = id.i12[m, n]
                                   state1M.initialState[id.ai1M[mt, n]] += dynamics[id.ai2M[mt, mn]] * mE[m]
                              end
                              if n < m
                                   nm = id.i12[n, m]
                                   state1M.initialState[id.ai1M[mt, n]] += dynamics[id.ai2M[mt, nm]] * mE[m]
                              end
                         end
                    end
               end

          elseif pr.particleType == "fermion"

               for mt in 1:nd.nmultp1M
                    for n in 1:nd.nsite1M
                         for m in 1:nd.nsite1M
                              if m < n
                                   mn = id.i12[m, n]
                                   state1M.initialState[id.ai1M[mt, n]] += dynamics[id.ai2M[mt, mn]] * mE[m]
                              end
                              if n < m
                                   nm = id.i12[n, m]
                                   state1M.initialState[id.ai1M[mt, n]] += dynamics[id.ai2M[mt, nm]] * mE[m]
                              end
                         end
                    end
               end

          end

          for mt in 1:nd.nmultp1M
               state1M.initialState[id.li1M[mt, :, :]] = dynamics[id.li2M[mt, :, :]]
          end

          state1M.norm = getStateVectorNorm(state1M.initialState, 1, model)

          return state1M
     end
end


function overlapAt(t::Float64, stateL::stateStruct, stateR::stateStruct, model::modelStruct)

     @assert stateL.manifold == stateR.manifold "Trying to overlap states of different manifolds."

     if stateL.manifold == stateR.manifold

          xL = (stateL.tfinal == stateL.tinitial) ? stateL.initialState : stateL.dyn(t)
          caL = getAlpha(xL, stateL.manifold, model)
          clL = getLambda(xL, stateL.manifold, model)

          xR = (stateR.tfinal == stateR.tinitial) ? stateR.initialState : stateR.dyn(t)
          caR = getAlpha(xR, stateR.manifold, model)
          clR = getLambda(xR, stateR.manifold, model)

          S = getS(clL, clR)

          return @tullio O := conj(caL[i, n]) * caR[j, n] * S[i, j]

     else
          error("Trying to overlap states of different manifolds.")
     end
end

function overlapAt(t::Float64, stateL::stateStruct, sVecR::Array{ComplexF64}, m::Int, model::modelStruct)

     @assert stateL.manifold == m "Trying to overlap states of different manifolds."

     xL = (stateL.tfinal == stateL.tinitial) ? stateL.initialState : stateL.dyn(t)
     caL = getAlpha(xL, stateL.manifold, model)
     clL = getLambda(xL, stateL.manifold, model)

     caR = getAlpha(sVecR, m, model)
     clR = getLambda(sVecR, m, model)

     S = getS(clL, clR)

     return @tullio O := conj(caL[i, n]) * caR[j, n] * S[i, j]
end

function overlapAt(t::Float64, sVecL::Array{ComplexF64}, stateR::stateStruct, m::Int, model::modelStruct)

     @assert stateR.manifold == m "Trying to overlap states of different manifolds."

     caL = getAlpha(sVecL, m, model)
     clL = getLambda(sVecL, m, model)

     xR = (stateR.tfinal == stateR.tinitial) ? stateR.initialState : stateR.dyn(t)
     caR = getAlpha(xR, stateR.manifold, model)
     clR = getLambda(xR, stateR.manifold, model)

     S = getS(clL, clR)

     return @tullio O := conj(caL[i, n]) * caR[j, n] * S[i, j]
end

function expValueAt(t::Float64, stateL::stateStruct, operator::transitionDipoleOperator, stateR::stateStruct, model::modelStruct)

     if operator.dir == "R"
          return overlapAt(t, stateL, getStateDynamicsAt(t, operator, stateR, model), (stateR.manifold + operator.action), model)
     elseif operator.dir == "L"
          return overlapAt(t, getStateDynamicsAt(t, operator, stateL, model), stateR, (stateL.manifold + operator.action), model)
     else
          error("Undefined operator action.")
     end
end

function getStateDynamicsAt(t::Float64, operator::transitionDipoleOperator, state::stateStruct, model::modelStruct)

     @unpack nd, id, pr = model

     mE = pr.dipGE * operator.polarizationVector

     dynamics = (state.tfinal == state.tinitial) ? state.initialState : state.dyn(t)

     if state.manifold == 0

          @assert operator.action == +1 "Undefined operator action."

          sVec = Vector{ComplexF64}(undef, nd.statesize1M)

          for mt in 1:nd.nmultp1M
               for n in 1:nd.nsite1M
                    sVec[id.ai1M[mt, n]] = dynamics[id.ai0M[mt, 1]] * mE[n]
               end
               sVec[id.li1M[mt, :, :]] = dynamics[id.li0M[mt, 1, :]]
          end

          return sVec

     elseif state.manifold == 1

          if operator.action == +1

               sVec = Vector{ComplexF64}(undef, nd.statesize2M)

               if pr.particleType == "boson"
                    for mt in 1:nd.nmultp2M
                         for n in 1:nd.nsite1M
                              nn = id.i12[n, n]
                              sVec[id.ai2M[mt, nn]] = sqrt(2.0) * dynamics[id.ai1M[mt, n]] * mE[n]
                              for m in 1:nd.nsite1M
                                   if n < m
                                        nm = id.i12[n, m]
                                        sVec[id.ai2M[mt, nm]] = (dynamics[id.ai1M[mt, n]] * mE[m] + dynamics[id.ai1M[mt, m]] * mE[n])
                                   end
                              end
                         end
                    end
               elseif pr.particleType == "fermion"
                    for mt in 1:nd.nmultp2M
                         for n in 1:nd.nsite1M
                              for m in 1:nd.nsite1M
                                   if n < m
                                        nm = id.i12[n, m]
                                        sVec[id.ai2M[mt, nm]] = (dynamics[id.ai1M[mt, n]] * mE[m] + dynamics[id.ai1M[mt, m]] * mE[n])
                                   end
                              end
                         end
                    end
               end

               for mt in 1:nd.nmultp2M
                    sVec[id.li2M[mt, :, :]] = dynamics[id.li1M[mt, :, :]]
               end

               return sVec

          elseif operator.action == -1

               sVec = zeros(ComplexF64, nd.statesize0M)

               for mt in 1:nd.nmultp0M
                    for n in 1:nd.nsite1M
                         sVec[id.ai0M[mt, 1]] += dynamics[id.ai1M[mt, n]] * mE[n]
                    end
                    sVec[id.li0M[mt, 1, :]] = dynamics[id.li1M[mt, :, :]]
               end

               return sVec

          else
               error("Undefined operator action.")
          end

     elseif state.manifold == 2

          @assert operator.action == -1 "Undefined operator action."

          sVec = zeros(ComplexF64, nd.statesize1M)

          if pr.particleType == "boson"
               for mt in 1:nd.nmultp1M
                    for n in 1:nd.nsite1M
                         nn = id.i12[n, n]
                         sVec[id.ai1M[mt, n]] += sqrt(2.0) * dynamics[id.ai2M[mt, nn]] * mE[n]
                         for m in 1:nd.nsite1M
                              if m < n
                                   mn = id.i12[m, n]
                                   sVec[id.ai1M[mt, n]] += dynamics[id.ai2M[mt, mn]] * mE[m]
                              end
                              if n < m
                                   nm = id.i12[n, m]
                                   sVec[id.ai1M[mt, n]] += dynamics[id.ai2M[mt, nm]] * mE[m]
                              end
                         end
                    end
               end
          elseif pr.particleType == "fermion"
               for mt in 1:nd.nmultp1M
                    for n in 1:nd.nsite1M
                         for m in 1:nd.nsite1M
                              if m < n
                                   mn = id.i12[m, n]
                                   sVec[id.ai1M[mt, n]] += dynamics[id.ai2M[mt, mn]] * mE[m]
                              end
                              if n < m
                                   nm = id.i12[n, m]
                                   sVec[id.ai1M[mt, n]] += dynamics[id.ai2M[mt, nm]] * mE[m]
                              end
                         end
                    end
               end
          end
          for mt in 1:nd.nmultp1M
               sVec[id.li1M[mt, :, :]] = dynamics[id.li2M[mt, :, :]]
          end

          return sVec

     end
end
