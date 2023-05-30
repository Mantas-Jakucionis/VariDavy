function δ(i,j)
     return Int(i==j)
end

function getReorgEnergy(J, wmin, wmax, p) # Frequency Input/output in [cm-1]
     wmin *= 0.000188365156731
     wmax *= 0.000188365156731
     return quadgk(w -> w*J(w, p), wmin, wmax, rtol=1e-8)[1] / 0.000188365156731  
 end

 function initializeBathReorg!(input)
     input.rsite .= getReorgEnergy(input.bathSpectralDensity, input.wmin, input.wmax, input.bathSpdParameters)
 end
  
function getStateLabels2M(model::modelStruct)
     return reshape(string.(model.id.i21),1,model.nd.nsite2M)
 end
 
function cdIntoSaveFolder(saveName::String)

     forbidden = ["<", ">", ":", "/", "?", "*", "."] 
     time = split(string(now()), "")
     for c in forbidden
          mask = time .== c
          time[mask] .= "_"
     end
     time = join(time)

     saveFolder = saveName * "_" * time
     mkdir(saveFolder)
     cd(saveFolder)    
 end

function diagramCutoff(M, cutoff = 1.0)
     ct = cutoff*maximum(abs.(M))
     replace!(x -> abs(x) > ct ? sign(x)*ct : x, M)
     return M
end

function diagramCutoffReal(M, cutoff = 1.0)
     x = diagramCutoff(real(M), cutoff)
     return x./maximum(abs.(x))
end

function diagramScaledReal(M, scaleValue)
     return diagramScaled(real(M), scaleValue)
end

function diagramScaled(M, scaleValue)
     Mabs = abs.(M)
     MabsMaxIndex = argmax(Mabs)
     MabsMaxSign = sign(M[MabsMaxIndex])
     @assert MabsMaxSign == sign(scaleValue) "Trying to normalize + and - values."
     return M ./ abs(scaleValue)
end

function processDiagram(t1::Vector{Float64}, t3::Vector{Float64}, input::Matrix{ComplexF64}, tau::Real = -1.0)
     data = similar(input)
     if tau != -1.0 && tau != 0.0 && tau != 0
         for i1 in 1:length(t1)
             for i3 in 1:length(t3)
                 data[i1,i3] = input[i1,i3] * exp(-(t1[i1]+t3[i3])/tau)
             end
         end
     else
         data = input
     end

     dataW = FFTW.fftshift(FFTW.fft(data, (1,2)))
     fs2cm = 5308.837458872913
     w1 = 2.0*pi*fs2cm*FFTW.fftshift(FFTW.fftfreq(length(t1), 1/(t1[2]-t1[1])))
     w3 = 2.0*pi*fs2cm*FFTW.fftshift(FFTW.fftfreq(length(t3), 1/(t3[2]-t3[1])))
     w3 = round.(w3)
     w1 = round.(w1)

     return w1, w3, dataW
 end

function plot2D(signal::Matrix{ComplexF64}, type::String, zmax::Float64, levels::Int, time2D; conv::Float64 = -1.0, xrange=-1, yrange=-1, title::String = "")

     if type != "2DES"
          w1, w3, pSignal = processDiagram(time2D, time2D, signal, conv);

          wxrange = (minimum(w1), maximum(w1))
          wyrange = (minimum(w3), maximum(w3))
          xlimits = xrange == -1 ? wxrange : xrange
          ylimits = yrange == -1 ? wyrange : yrange 

          if type == "R" || type == "r"
               wyrange =  -1 .* ylimits
          else
               wyrange = ylimits
          end
          pSignal = transpose(diagramCutoffReal(pSignal, zmax))
          x = contourf(w1, w3, pSignal, levels = -1.0:(2/levels):1.0, fillcolor=cgrad([:blue, :green, :white, :orange, :red], [0.0, 0.5, 1]), aspect_ratio=:equal);
          x = plot!( [wxrange...], [wyrange...], color="black", xlabel=L"\mathrm{\omega_{1}, \ cm^{-1}}", ylabel=L"\mathrm{\omega_{3}, \ cm^{-1}}", title=title, xlims=xlimits, ylims=ylimits, legend=false)
     end
end

function plot2D(diag_R::Matrix{ComplexF64}, diag_nR::Matrix{ComplexF64}, type::String, zmax::Float64, levels::Int, time2D; conv::Float64=-1.0, xrange=-1, yrange=-1, title::String = "")

     if type == "2DES" || type == "ES" || type == "2D" || type == "2d"
          w1, w3, pDiag_R = processDiagram(time2D, time2D, diag_R, conv);
          w1, w3, pDiag_nR = processDiagram(time2D, time2D, diag_nR, conv);

          wxrange = (minimum(w1), maximum(w1))
          wyrange = (minimum(w3), maximum(w3))
          xlimits = xrange == -1 ? wxrange : xrange
          ylimits = yrange == -1 ? wyrange : yrange 
          pSignal = pDiag_nR + reverse(pDiag_R, dims=1)
          pSignal = transpose(diagramCutoffReal(pSignal, zmax))
          x = contourf(w1, w3, pSignal, levels = -1.0:(2/levels):1.0, fillcolor=cgrad([:blue, :green, :white, :orange, :red], [0.0, 0.5, 1]), aspect_ratio=:equal);
          x = plot!( [wxrange...], [wyrange...], color="black", xlabel=L"\mathrm{\omega_{1}, \ cm^{-1}}", ylabel=L"\mathrm{\omega_{3}, \ cm^{-1}}", title=title, xlims=xlimits, ylims=ylimits, legend=false)
     end
end
 

function getS(bra, ket)
     @tullio S[i,j] := exp <| conj(bra[i,n,h])*ket[j,n,h] - 0.5*abs2(bra[i,n,h]) - 0.5*abs2(ket[j,n,h])
end  
 
function getD(kap, lam)
     @tullio D[j] := exp <| conj(kap[j][r])*lam[j][r] - 0.5*abs2(kap[j][r]) - 0.5*abs2(lam[j][r])
end  
 
function getDistance(bra, ket, model::modelStruct)
     @unpack Dmt = model.id
     @tullio dist[i,j] := sqrt <| abs2(bra[i,a,b]*Dmt[j] - ket[j,a,b]*Dmt[i])
end
 
function getModeDisplacement(w::Real, temp::Real, model::modelStruct)
     @unpack pr = model

    if w != 0.0
         sigma = sqrt( 1.0/(exp(w/(kbfs*temp)) - 1.0)/2.0 )
         return sqrt(2.0)*rand(Normal(0,sigma))*exp(im*rand()*2.0*pi)
    else
         return ComplexF64(0.0, 0.0)
    end
end

function getAlpha(x::Vector{ComplexF64}, m::Int, model::modelStruct)
     @unpack nd, id = model
    if m == 0
         return @views x[id.ai0M]
    elseif m == 1
         return @views x[id.ai1M]
    elseif m == 2
         return @views x[id.ai2M]
    end
end

function getLambda(x::Vector{ComplexF64}, m::Int, model::modelStruct)
     @unpack nd, id = model
    if m == 0
         return @views x[id.li0M]
    elseif m == 1
         return @views x[id.li1M]
    elseif m == 2
         return @views x[id.li2M]
    end
end

function initializeEnsembleStaticEnergyShifts!(model::modelStruct)
     @unpack pr, nd, ens = model
     ens.dE = map(_ -> rand.(map(n -> Normal(0.0, pr.staticDisorderSigma*cm2fs), 1:nd.nsite1M)), 1:nd.ntraj)
     return nothing
end
 
function initializeEnsembleExternalFieldPolarizations!(model::modelStruct)
     @unpack nd, pr, ens = model
     ens.eVec = map(_ -> begin
          eVec = Array{Float64}(undef, 3)
          if pr.extFieldDistribution == "static"
               eVec = pr.extEstatic
          elseif pr.extFieldDistribution == "uniform"
               eVec = randn(3).*2 .- 1
               eVec ./= sqrt(sum(eVec.*eVec))
          else
               error("Unknown 'extFieldDistribution' input variable value: $(pr.extFieldDistribution)", )
          end

     end, 1:nd.ntraj)

     return nothing
 end
 
function initializeEnsembleInitialConditions!(model::modelStruct)
     @unpack nd, id, pr, ens = model
     @unpack iwG, temp = pr

     iwG = iwG[:]

     ens.u0 = map( _ -> begin

          x = zeros(ComplexF64, nd.statesize0M)
          
          mt = 1
          for n in 1:nd.nsite0M
               x[id.ai0M[mt,n]] = ComplexF64(1.0, 0.0)
          end

          for n in 1:nd.nbath0M
               for h in 1:nd.nmode0M
                    x[id.li0M[mt,n,h]] = getModeDisplacement(iwG[h], temp, model)
               end
          end

          setUnpopulatedMultiples!(x, 0, model)

          return getNormalizedStateVector0M(x, model)
     end, 1:nd.ntraj)

     return nothing
end

function getThermalizationData(model::modelStruct; trfTraj::Bool = false)

     @unpack pr, nd = model

     thermProb = pr.thermStep * pr.thermRate
     @assert thermProb <= 1.0 "Thermalization probability is more than 1."

     thermTimes = trfTraj ? collect(pr.tmin+pr.thermStep:pr.thermStep:pr.trfThermTime) : collect(pr.tmin+pr.thermStep:pr.thermStep:pr.tmax)

     if !trfTraj
          data =  map( _ -> begin

               thermSize = 0

               if model.pr.thermType == "mode"
                    thermSize = nd.nmode0M
               elseif model.pr.thermType == "bath"
                    thermSize = nd.nbath1M
               elseif model.pr.thermType == "global"
                    thermSize = 1
               else
                    error("Unknown thermType value.")
               end

               events = rand(Binomial(1, thermProb), length(thermTimes), thermSize)
               
               if model.pr.thermType == "bath"
                    e = Array{Int64}(undef, length(thermTimes), nd.nmode0M)
                    for t in 1:length(thermTimes), i in 1:thermSize
                         e[t, 1 + (i-1)*nd.nmode1M : i*nd.nmode1M] .= events[t,i]
                    end
                    events = e
               elseif model.pr.thermType == "global"
                    e = Array{Int64}(undef, length(thermTimes), nd.nmode0M)
                    for t in 1:length(thermTimes)
                         e[t, :] .= events[t,1]
                    end
                    events = e
               end

               stopMask = sum(events, dims=2)[:] .!= 0
               
               times = thermTimes[stopMask]
               events = events[stopMask,:]

               return thermStruct(times, events)
          
          end, 1:nd.ntraj)
     else
          thermSize = 0

          if model.pr.thermType == "mode"
               thermSize = nd.nmode0M
          elseif model.pr.thermType == "bath"
               thermSize = nd.nbath1M
          elseif model.pr.thermType == "global"
               thermSize = 1
          else
               error("Unknown thermType value.")
          end

          events = rand(Binomial(1, thermProb), length(thermTimes), thermSize)
          
          if model.pr.thermType == "bath"
               e = Array{Int64}(undef, length(thermTimes), nd.nmode0M)
               for t in 1:length(thermTimes), i in 1:thermSize
                    e[t, 1 + (i-1)*nd.nmode1M : i*nd.nmode1M] .= events[t,i]
               end
               events = e
          elseif model.pr.thermType == "global"
               e = Array{Int64}(undef, length(thermTimes), nd.nmode0M)
               for t in 1:length(thermTimes)
                    e[t, :] .= events[t,1]
               end
               events = e
          end

          stopMask = sum(events, dims=2)[:] .!= 0
          
          times = thermTimes[stopMask]
          events = events[stopMask,:]

          return thermStruct(times, events)
     end

     return data
end

function initializeEnsembleThermalizationEvents!(model::modelStruct)

     @unpack pr, nd, ens = model

     thermProb = pr.thermStep * pr.thermRate
     @assert thermProb <= 1.0 "Thermalization probability is more than 1."
     
     if model.pr.thermalization == 1
          ens.therm = getThermalizationData(model)
     else
          ens.therm = map( _ -> thermStruct(0), 1:nd.ntraj)
     end

     ens.thermTrf = map( _ -> getThermalizationData(model; trfTraj = true), 1:nd.ntraj)
end

function setUnpopulatedMultiples!(stateVector::Vector{ComplexF64}, manifold::Int, model::modelStruct)

     @unpack nd, id, pr = model

     if manifold == 0
          mt_range   = nd.nmultp0M
          bath_range = nd.nbath0M
          mode_range = nd.nmode0M
          li = id.li0M
     elseif manifold == 1
          mt_range   = nd.nmultp1M
          bath_range = nd.nbath1M
          mode_range = nd.nmode1M
          li = id.li1M
     elseif manifold == 2
          mt_range   = nd.nmultp2M
          bath_range = nd.nbath2M
          mode_range = nd.nmode2M
          li = id.li2M
     else
          error("Wrong manifold number.")
     end

     mt_range > 1 ? converged = 0 : converged = 1

     s = pr.multpDistance

     cnt = 0
     while converged == 0 

          nIW = zeros(Int, bath_range, mode_range)
          
          if pr.multpDistr == "grid"
               nIW .+= 1
               dx = 1.0
               dmin = -10.0
               dmax = 10.0
               x = dmin+dx/2:dx:dmax-dx/2
               
               @tullio  grid[i,j] := $s * (x[i] + im*x[j]) 
               grid = grid[sortperm(abs.(grid[:]))]
          end

          for mt in 2:mt_range

               for n in 1:bath_range
                    for h in 1:mode_range

                         if pr.multpDistr == "grid"
                              stateVector[li[mt,n,h]] = stateVector[li[1,n,h]] + grid[nIW[n,h]]
                         elseif pr.multpDistr == "polygon"
                              perLayer = pr.multpPerLayer
                              fl = floor(nIW[n,h]/perLayer)
                              stateVector[li[mt,n,h]] = stateVector[li[1,n,h]] + s * (1 + fl) * exp(im*2*pi/perLayer*nIW[n,h] + im*2*pi/perLayer/2 * floor(nIW[n,h]/perLayer))
                         else
                              error("Unknown multiplicity distribution.")
                         end
                         
                         nIW[n,h] += 1
                    end
               end     
          end

          cl0M = getLambda(stateVector, manifold, model)
          S0M   = getS(cl0M, cl0M)
                    
          all( abs.(S0M)[1,2:end] .> pr.d2Thres ) ? converged = 1 : s *= 0.9
          
          cnt += 1
     end
     
     if mt_range > 1
          cl0M = getLambda(stateVector, manifold, model)
          S0M   = getS(cl0M, cl0M)
          @assert any(getDistance(cl0M, cl0M, model)[1,2:end] .>= pr.apopThres) "Initial overlap is too high."
     end
     return nothing
end

function getStateVectorEnergy1M(stateVector, model::modelStruct; normAmp::Float64=1.0)
 
     x = Array(reinterpret(ComplexF64, stateVector))

     ca  = getAlpha(x, 1, model)
     cl  = getLambda(x, 1, model)
     S   = getS(cl, cl)
 
     @unpack H, iwG, iwE, f, diw = model.pr
     @unpack Dmt = model.id
 
     @tullio  G[i,j,n,m] := conj(ca[i,n])*ca[j,m]*S[i,j]
     @tullio  P[i,j,n]   := G[i,j,n,n]
     
     @tullio  elec    :=  G[i,j,n,m]*H[n,m]
     @tullio  vib     :=  P[i,j,n]*iwG[k,h]*conj(cl[i,k,h])*cl[j,k,h]
     @tullio  elvib   := -P[i,j,n]*iwE[n,h]*f[n,h]*( conj(cl[i,n,h])*Dmt[j] + cl[j,n,h]*Dmt[i] )
     @tullio  elvib2  :=  P[i,j,n]*diw[n,h]*( 1.0 + (conj(cl[i,n,h])*Dmt[j] + cl[j,n,h]*Dmt[i])^2 ) /4.0 
     @tullio  pop     :=  conj(ca[i,n])*ca[j,n]*S[i,j]

     energy = real(elec+vib+elvib+elvib2) / real(pop) * normAmp
     
     return energy
 end
 
function getStateVectorEnergy1M_MTP1(stateVector, model::modelStruct; normAmp::Float64=1.0)
 
     x = Array(reinterpret(ComplexF64, stateVector))

     ca = @views x[model.id.ai1M_MTP1]
     cl = @views x[model.id.li1M_MTP1]
 
     @unpack H, iwG, iwE, f, diw = model.pr
     @unpack Dmt = model.id
 
     @tullio  G[n,m] := conj(ca[n])*ca[m]
     @tullio  P[n]   := G[n,n]
     
     @tullio  elec    :=  G[n,m]*H[n,m]
     @tullio  vib     :=  P[n]*iwG[k,h]*conj(cl[k,h])*cl[k,h]
     @tullio  elvib   := -P[n]*iwE[n,h]*f[n,h]*( conj(cl[n,h])+ cl[n,h] )
     @tullio  elvib2  :=  P[n]*diw[n,h]*( 1.0 + (conj(cl[n,h]) + cl[n,h])^2 ) /4.0  
     @tullio  pop     :=  P[n]

     energy = real(elec+vib+elvib+elvib2) / real(pop) * normAmp
     
     return energy
 end
 

function findMinimalEnergyStateVector1M(state::stateStruct, model::modelStruct)
     
     initialStateVector = Array(reinterpret(Float64, state.initialState))
     
     lbound = Array( reinterpret(Float64, vcat(
          ComplexF64.(-1*ones(Float64, model.nd.astatesize1M), -1*ones(Float64, model.nd.astatesize1M)),
          ComplexF64.(-20*ones(Float64, model.nd.lstatesize1M), -20*ones(Float64, model.nd.lstatesize1M))
          )))
          
     ubound = -1 .* lbound

     f = OptimizationFunction(getStateVectorEnergy1M)
     prob = Optimization.OptimizationProblem(f, initialStateVector, model, lb=lbound, ub=ubound)

     optimalStateVector = solve(prob, ParticleSwarm(lower=prob.lb, upper= prob.ub))
     
     complexOptimalStateVector = Array(reinterpret(ComplexF64, optimalStateVector))
     optimalStateVector = Array(reinterpret(Float64, complexOptimalStateVector))
     
     optimalEnergy = getStateVectorEnergy1M(optimalStateVector, model)
     initialEnergy = getStateVectorEnergy1M(initialStateVector, model)
     
     println("Optimization dE: ", (optimalEnergy - initialEnergy) * fs2cm)

     return complexOptimalStateVector
end

function findMinimalEnergyStateVector1M_MTP1(state::stateStruct, model::modelStruct)

     println("Minimal energy: ", optimalEnergy * fs2cm)
     @unpack id = model

     ca = state.initialState[id.ai1M[1,:]]
     cl = state.initialState[id.li1M[1,:,:]]

     initialState = vcat(ca[:],cl[:])

     initialStateVector_MTP1 = Array(reinterpret(Float64, initialState))
     
     casize, clsize = length.([ca,cl])

     lbound = Array( reinterpret(Float64, vcat(
          ComplexF64.( -1*ones(Float64, casize), -1*ones(Float64, casize)),
          ComplexF64.(-20*ones(Float64, clsize), -20*ones(Float64, clsize))
          )))
          
     ubound = -1 .* lbound

     f = OptimizationFunction(getStateVectorEnergy1M_MTP1)
     prob = Optimization.OptimizationProblem(f, initialStateVector_MTP1, model, lb=lbound, ub=ubound)

     optimalStateVector_MTP1 = solve(prob, ParticleSwarm(lower=prob.lb, upper= prob.ub))
     
     state_MTP1 = Array(reinterpret(ComplexF64, optimalStateVector_MTP1))

     complexOptimalStateVector = deepcopy(state.initialState)
     complexOptimalStateVector[id.ai1M[1,:]]   = state_MTP1[id.ai1M_MTP1]
     complexOptimalStateVector[id.li1M[1,:,:]] = state_MTP1[id.li1M_MTP1]
     setUnpopulatedMultiples!(complexOptimalStateVector, 1, model)
          
     optimalEnergy = getStateVectorEnergy1M_MTP1(optimalStateVector_MTP1, model)
     initialEnergy = getStateVectorEnergy1M_MTP1(initialStateVector_MTP1, model)

     println("Initial energy: ", initialEnergy * fs2cm)
     println("Minimal energy: ", optimalEnergy * fs2cm)
     println("Optimization dE: ", (optimalEnergy - initialEnergy) * fs2cm)
 
     return complexOptimalStateVector
 end

 function plotCoherentStateDynamics(state::stateStruct, model::modelStruct, dt::Real, fpsList::Vector{Int}, saveName::String = "test"; nb::Int = 1, nm::Int = 1)

     if state.manifold == 0
          nmtp = model.nd.nmultp0M
          nsite = model.nd.nsite0M
     elseif state.manifold == 1
          nmtp = model.nd.nmultp1M
          nsite = model.nd.nsite1M
     elseif state.manifold == 2
          nmtp = model.nd.nmultp2M
          nsite = model.nd.nsite2M
     end

     trange = state.tinitial:dt:state.tfinal
     ca  = map(t -> getAlpha(state.dyn(t), state.manifold, model), trange);
     cl  = map(t -> getLambda(state.dyn(t), state.manifold, model), trange);
     cix = real.(cl) .* sqrt(2.0);
     cip = imag.(cl) .* sqrt(2.0);
 
     S = map(t -> getS(cl[t],cl[t]), 1:length(trange));
 
     N = zeros(ComplexF64, length(trange), nmtp)
     for t in 1:length(trange)
         for n in 1:nsite
             for k in 1:nmtp
                 for i in 1:nmtp
                     for j in 1:nmtp
                         if (k != i) || (k != j)
                             N[t,k] += conj(ca[t][i,n])*ca[t][j,n]*S[t][i,j];
                         end
                     end
                 end
             end
         end
     end
     N = real.(N)
     N ./= maximum(N)

     d(i,j) = Int(i==j)
     
     d1(k,i,j) = Int( (k != i) || (k != j) )
     @tullio  N1[t,k] := real <| conj(ca[t][i,n])*ca[t][j,n]*S[t][i,j]*d1(k,i,j) (k in 1:nmtp)
     N1 ./= maximum(N1)    
     
     d2(k,l,i,j) = d1(k,i,j) * d1(l,i,j)
     @tullio  N2[t,k,l] := real <| conj(ca[t][i,n])*ca[t][j,n]*S[t][i,j]*d2(k,l,i,j) (k in 1:nmtp, l in 1:nmtp)
     N2 ./= maximum(N2)   
     
     @tullio  dN12[t,k,l] := (1-N2[t,k,l]) - ((1 - N2[t,k,k]) + (1-N2[t,l,l]))

     xmin = -5
     xmax = 10
     dx = 0.1
     xrange = xmin:dx:xmax
     nfun = 10
     
     H = getHermiteFunctionSet(nfun)
     
     F = map( n -> H[n].(xrange).*exp.(-0.5.*xrange.^2) ./ sqrt( 2^n * factorial(n) * sqrt(pi) ) , 0:(nfun-1) )
     F = OffsetArray(F, 0:(nfun-1));
     
     @tullio  Sm[t,i,j] := conj(cl[t][i,n,h])*cl[t][j,n,h]
     @tullio  Sm[t,i,j] += -0.5*conj(cl[t][i,n,h])*cl[t][i,n,h]
     @tullio  Sm[t,i,j] += -0.5*conj(cl[t][j,n,h])*cl[t][j,n,h]

     @tullio Sm_diff[t,i,j,k,q] := Sm[t,i,j] - conj(cl[t][i,k,q])*cl[t][j,k,q]
     map!(exp, Sm_diff, Sm_diff)
     
     @tullio  C[t,i,j,k,q,x] := conj(cl[t][i,k,q]).^u .* cl[t][j,k,q].^v .* F[u][x] .* F[v][x] ./ sqrt(factorial(u)*factorial(v))
     @tullio  P[t,k,q,x] := real <| conj(ca[t][i,n])*ca[t][j,n]*Sm_diff[t,i,j,k,q]*C[t,i,j,k,q,x]

     P_norm = sum(P, dims=4)

     @tullio Pn[t,k,q,x] := P[t,k,q,x] / P_norm[t,k,q,1] * 25

     if size(N, 2) == 1
          N .= 0.0
     end
     
     anim1 = @animate for ti in 1:length(trange)
 
         scatter((cix[ti][1,nb,nm], cip[ti][1,nb,nm]), xlims=[xmin, 5], ylims=[-5, 5], title= "time: " * string(trange[ti]) * " fs", label="Gaussian #1",
         markersize = 10,
         markeralpha = tanh(1.0 - N[ti,1])/tanh(1.0),
         markercolor = :red, 
         markerstrokewidth = 2,
         markerstrokealpha = 1,
         markerstrokecolor = :black,
         xlabel = L"\mathrm{Coordinate, x_{1,1}}",
         ylabel = L"\mathrm{Momentum, p_{1,1}}")
 
         for i in 2:nmtp
             scatter!((cix[ti][i,nb,nm], cip[ti][i,nb,nm]), label="Gaussian #$(i)",
             markersize = 10,
             markeralpha = tanh(1.0 - N[ti,i])/tanh(1.0),
             markercolor = :red, 
             markerstrokewidth = 2,
             markerstrokealpha = 1,
             markerstrokecolor = :black)
         end

         plot!(xrange, Pn[ti,nb,nm,:], label=L"$\mathrm{Probability, P(x_{1,1})}$",
               linewidth = 2, 
               legend=:outertopright,
               legendfontsize=6,
               framestyle = :box);

     end

     for i in 1:length(fpsList)
         gif(anim1, "$(saveName)_bath$(nb)_mode$(nm)_fps$(fpsList[i]).gif", fps = fpsList[i])
     end
 
 end

function getHermiteFunctionSet(nfun::Int)
     Hermite = OffsetArray(Array{Function}(undef, nfun), 0:nfun-1)
     Hermite[0] = x -> 1.0
     Hermite[1] = x -> 2.0*x
     for np in 2:(nfun+Hermite.offsets[1])
         n = np - 1
         Hermite[np] = x -> 2*x*Hermite[n](x) - 2*n*Hermite[n-1](x)
     end
     return Hermite
 end
