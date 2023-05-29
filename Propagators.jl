function D2_2M!(dx::Vector{ComplexF64}, x::Vector{ComplexF64}, model::modelStruct, t::Float64)

    @unpack nd, id, pr = model
    @unpack H, iwG, iwE, f, K, diw = pr
    @unpack ai2M, li2M, N1, N2 = id

    ca = getAlpha(x, 2, model)
    cl = getLambda(x, 2, model)

    @tullio  rho = abs2(ca[1,nm])
    @tullio  N1d[v,ab,nm] := ca[1,ab]*N1[v,kl,v,nm]*conj(ca[1,kl]) / rho

    @tullio  tt[v] := iwE[v,h]*f[v,h]*real(cl[1,v,h])
    
    @tullio  dx[ai2M[1,ab]] =  -N1[v,ab,u,nm]*ca[1,nm]*H[v,u]
    @tullio  dx[ai2M[1,ab]] += +ca[1,nm]*N1[v,ab,v,nm]*2.0*tt[v]
    @tullio  dx[ai2M[1,ab]] += -ca[1,nm]*0.5*N1d[v,ab,nm]*2.0*tt[v]
    # @tullio  dx[ai2M[1,ab]] += -N2[v,u,ab,nm]*ca[1,nm]*K[v,u] # Enable for double excitation shifts
    
    @tullio  dx[li2M[1,k,h]]  = -iwG[k,h]*cl[1,k,h]
    @tullio  dx[li2M[1,k,h]] += +iwE[k,h]*N1d[k,nm,nm]*f[k,h]
    
    model.pr.anharmQ ? begin
        @tullio  dx[ai2M[1,ab]] += -ca[1,nm]*N1[v,ab,v,nm]*diw[v,h]*(1.0 + 4.0*real(cl[1,v,h])^2) /4.0 
        @tullio  dx[ai2M[1,ab]] += +ca[1,nm]*N1d[v,ab,nm]*diw[v,h]*4.0*real(cl[1,v,h])^2 /4.0 
        
        @tullio  dx[li2M[1,k,h]] += -diw[k,h]*N1d[k,nm,nm]*4.0*real(cl[1,k,h])/4.0 
    end : nothing

    dx .*= im
    
    return nothing
end

function D2_1M!(dx::Vector{ComplexF64}, x::Vector{ComplexF64}, model::modelStruct, t::Float64)

    @unpack nd, id, pr = model
    @unpack H, iwG, iwE, f, diw = pr
    @unpack ai1M, li1M = id

    ca = getAlpha(x, 1, model)
    cl = getLambda(x, 1, model)
    
    @tullio  rho = abs2(ca[1,n])
    @tullio  F := abs2(ca[1,m])*iwE[m,h]*f[m,h]*real(cl[1,m,h]) / rho
    
    @tullio  dx[ai1M[1,n]] =  -ca[1,m]*H[n,m]
    @tullio  dx[ai1M[1,n]] += -ca[1,n]*F
    @tullio  dx[ai1M[1,n]] += +ca[1,n]*iwE[n,h]*f[n,h]*2.0*real(cl[1,n,h])
    
    @tullio  dx[li1M[1,k,h]]  = -iwG[k,h]*cl[1,k,h]
    @tullio  dx[li1M[1,k,h]] += +iwE[k,h]*abs2(ca[1,k])*f[k,h] / rho
    
    model.pr.anharmQ ? begin
        @tullio  D := abs2(ca[1,m])*diw[m,q]*real(cl[1,m,q])^2 / rho

        @tullio  dx[ai1M[1,n]] += -ca[1,n]*diw[n,h]*(0.25 + real(cl[1,n,h])^2)
        @tullio  dx[ai1M[1,n]] += +ca[1,n]*D

        @tullio  dx[li1M[1,k,h]] += -diw[k,h]*abs2(ca[1,k])*real(cl[1,k,h]) / rho
    end : nothing

    dx .*= im

    return nothing
end

function D2_0M!(dx::Vector{ComplexF64}, x::Vector{ComplexF64}, model::modelStruct, t::Float64)

    @unpack nd, id, pr = model
    @unpack iwG = pr
    @unpack ai0M, li0M = id

    iwG = @view iwG[:]

    cl = getLambda(x, 0, model)

    @tullio  dx[ai0M[1,1]]   = 0.0
    @tullio  dx[li0M[1,1,h]] = -iwG[h]*cl[1,1,h]
    dx .*= im
    
    return nothing
end

function D2_0M_MTP1!(dx::Vector{ComplexF64}, x::Vector{ComplexF64}, model::modelStruct, t::Float64)

    @unpack nd, id, pr = model
    @unpack iwG = pr
    @unpack ai0M, li0M, li0M = id

    iwG = @view iwG[:]

    cl = getLambda(x, 0, model)

    @tullio  dx[ai0M[1,1]]   = 0.0
    @tullio  dx[li0M[j,1,h]] = -iwG[h]*cl[1,1,h]
    dx .*= im
    
    return nothing
end

function mD2_2M_ODE_RI_APO!(dx::Vector{ComplexF64}, x::Vector{ComplexF64}, cachePack::cachePackStructFloat64, t::Float64)

    @unpack model, M, RHS = cachePack;
    @unpack H, iwG, iwE, f, diw = model.pr;
    @unpack aRi2M, aIi2M, lRi2M, lIi2M, Dn2M, Dmt, N1, N2 = model.id;
    @unpack ai2M, li2M = model.id;
    @unpack iterRtol, apopThres = model.pr;

    ca = getAlpha(x, 2, model)
    cl = getLambda(x, 2, model)
    S = getS(cl, cl)
    
    Dist = getDistance(cl, cl, model)
    Dist = (Dist + (apopThres+1.0)*Diagonal(ones(Float64, model.nd.nmultp2M))) .< apopThres
    iDist = weakly_connected_components(SimpleGraph(Dist))
    iDist = iDist[length.(iDist) .> 1]
    apopQ = length(iDist) != 0 ? true : false 
    
    @tullio  P[i,nm,j,kl] := conj(ca[i,nm])*ca[j,kl]*S[i,j]
    @tullio  Pm[i,j,n] := conj(ca[i,n])*ca[j,n]*S[i,j] 
    @tullio  Pm[i,j,n] += 1e-7*S[i,j] 

    @tullio  JK[kl,nm] := N1[v,kl,u,nm]*H[v,u]
    # @tullio  JK[kl,nm] += N2[v,u,kl,nm]*K[v,u] # Enable for double excitation shifts

    @tullio  A1[i,j] := iwG[k,h]*conj(cl[i,k,h])*cl[j,k,h]
    @tullio  A2[i,j,k] := -iwE[k,h]*f[k,h]*(conj(cl[i,k,h])*Dmt[j] + cl[j,k,h]*Dmt[i])
    
    @tullio  RHS[ai2M[i,nm]] =  ca[j,kl]*S[i,j]*JK[nm,kl]
    @tullio  RHS[ai2M[i,nm]] += ca[j,kl]*S[i,j]*N1[v,nm,v,kl]*A2[i,j,v]
    @tullio  RHS[ai2M[i,nm]] += ca[j,nm]*S[i,j]*A1[i,j]
    
    @tullio  RHS[li2M[i,k,h]] =  P[i,kl,j,nm]*cl[j,k,h]*JK[kl,nm]
    @tullio  RHS[li2M[i,k,h]] += P[i,kl,j,nm]*cl[j,k,h]*N1[v,kl,v,nm]*A2[i,j,v]
    @tullio  RHS[li2M[i,k,h]] += P[i,nm,j,nm]*cl[j,k,h]*A1[i,j]
    @tullio  RHS[li2M[i,k,h]] += P[i,nm,j,nm]*iwG[k,h]*cl[j,k,h]
    @tullio  RHS[li2M[i,k,h]] += -P[i,kl,j,nm]*N1[k,kl,k,nm]*iwE[k,h]*f[k,h]

    model.pr.anharmQ ? begin
        @tullio  A3[i,j,k] := diw[k,h] * ( 1.0 + (conj(cl[i,k,h])*Dmt[j] + cl[j,k,h]*Dmt[i])^2 ) /4.0 
            
        @tullio  RHS[ai2M[i,nm]] += ca[j,kl]*S[i,j]*N1[v,nm,v,kl]*A3[i,j,v]

        @tullio  RHS[li2M[i,k,h]] += P[i,kl,j,nm]*cl[j,k,h]*N1[v,kl,v,nm]*A3[i,j,v]
        @tullio  RHS[li2M[i,k,h]] += P[i,kl,j,nm]*N1[k,kl,k,nm]*2.0*diw[k,h]*(conj(cl[i,k,h])*Dmt[j] + cl[j,k,h]*Dmt[i]) /4.0 
    end : nothing

    RHS .*= -im

    @tullio  iL[i,j,m,h] := conj(cl[i,m,h])*Dmt[j] - 0.5*conj(cl[j,m,h])*Dmt[i]
    @tullio  iLm[i,j,m,h] := iL[i,j,m,h] - 0.5*cl[j,m,h]*Dmt[i]
    @tullio  iLp[i,j,m,h] := iL[i,j,m,h] + 0.5*cl[j,m,h]*Dmt[i]

    M .= 0.0

    @tullio  M[aRi2M[i,n], aRi2M[j,n]] = real( S[i,j] )
    @tullio  M[aRi2M[i,n], aIi2M[j,n]] = real(    im*S[i,j] )
    @tullio  M[aRi2M[i,n], lRi2M[j,m,f]] = real(   ca[j,n]*S[i,j]*iLm[i,j,m,f] )
    @tullio  M[aRi2M[i,n], lIi2M[j,m,f]] = real( im*ca[j,n]*S[i,j]*iLp[i,j,m,f] )

    @tullio  M[aIi2M[i,n], aRi2M[j,n]] = imag(      S[i,j] )
    @tullio  M[aIi2M[i,n], aIi2M[j,n]] = imag(    im*S[i,j] )
    @tullio  M[aIi2M[i,n], lRi2M[j,m,f]] = imag(   ca[j,n]*S[i,j]*iLm[i,j,m,f] )
    @tullio  M[aIi2M[i,n], lIi2M[j,m,f]] = imag( im*ca[j,n]*S[i,j]*iLp[i,j,m,f] )

    @tullio  M[lRi2M[i,k,h], aRi2M[j,n]] = real(      conj(ca[i,n])*S[i,j]*cl[j,k,h]  )
    @tullio  M[lRi2M[i,k,h], aIi2M[j,n]] = real(    im*conj(ca[i,n])*S[i,j]*cl[j,k,h]  )
    @tullio  M[lRi2M[i,k,h], lRi2M[j,k,h]] = real(    Pm[i,j,n] )
    @tullio  M[lRi2M[i,k,h], lIi2M[j,k,h]] = real(  im*Pm[i,j,n] )
    @tullio  M[lRi2M[i,k,h], lRi2M[j,m,f]] += real(   Pm[i,j,n]*cl[j,k,h]*iLm[i,j,m,f] )
    @tullio  M[lRi2M[i,k,h], lIi2M[j,m,f]] += real( im*Pm[i,j,n]*cl[j,k,h]*iLp[i,j,m,f] )

    @tullio  M[lIi2M[i,k,h], aRi2M[j,n]] = imag(      conj(ca[i,n])*S[i,j]*cl[j,k,h]  )
    @tullio  M[lIi2M[i,k,h], aIi2M[j,n]] = imag(    im*conj(ca[i,n])*S[i,j]*cl[j,k,h]  )
    @tullio  M[lIi2M[i,k,h], lRi2M[j,k,h]] = imag(    Pm[i,j,n] )
    @tullio  M[lIi2M[i,k,h], lIi2M[j,k,h]] = imag(  im*Pm[i,j,n] )
    @tullio  M[lIi2M[i,k,h], lRi2M[j,m,f]] += imag(   Pm[i,j,n]*cl[j,k,h]*iLm[i,j,m,f] )
    @tullio  M[lIi2M[i,k,h], lIi2M[j,m,f]] += imag( im*Pm[i,j,n]*cl[j,k,h]*iLp[i,j,m,f] )

    apopQ ? begin 
        for key in iDist
            i = key[1]
            li2M_i = @view li2M[i,:,:]
            lRi2M_i = @view lRi2M[i,:,:]
            lIi2M_i = @view lIi2M[i,:,:]
            for k in 2:length(key)
                j = key[k]
                li2M_j = @view li2M[j,:,:]
                RHS[li2M_i] .+= RHS[li2M_j]
                RHS[li2M_j] .= -111.0
                lRi2M_j = @view lRi2M[j,:,:]
                lIi2M_j = @view lIi2M[j,:,:]
                M[ lRi2M_i, :] .+= M[ lRi2M_j, :]
                M[ lIi2M_i, :] .+= M[ lIi2M_j, :]
                M[:, lRi2M_i] .+= M[:, lRi2M_j]
                M[:, lIi2M_i] .+= M[:, lIi2M_j]
                M[lRi2M_j, :] .= -111.0
                M[lIi2M_j, :] .= -111.0
                M[:, lRi2M_j] .= -111.0
                M[:, lIi2M_j] .= -111.0
            end
        end

        iRHS = findall(RHS .== -111.0)
        setDiffRHS = setdiff(1:length(x), iRHS)
        RHS = @view RHS[setDiffRHS]

        iM = findall(M[1,:] .== -111.0)
        setDiffM = setdiff(1:size(M)[1], iM)
        M = @view M[setDiffM, setDiffM]

        dx .= ComplexF64(0.0, 0.0)
        dx[setDiffRHS] .= reinterpret(ComplexF64, gmres(M, reinterpret(Float64, RHS), verbose=false, reltol = iterRtol, restart = 200, Pl = lu(M) ))
   
        for key in iDist
            i = key[1]
            li2M_i = @view li2M[i,:,:]
            for k in 2:length(key)
                j = key[k]
                li2M_j = @view li2M[j,:,:]
                dx[li2M_j] .= dx[li2M_i]
            end
        end
    end : dx .= reinterpret(ComplexF64, gmres(M, reinterpret(Float64, RHS), verbose=false, reltol = iterRtol, restart = 200, Pl = lu(M) ))
    
    # println("2M: ", t)
    return nothing
end

function mD2_1M_ODE_RI_APO!(dx::Vector{ComplexF64}, x::Vector{ComplexF64}, cachePack::cachePackStructFloat64, t::Float64)

    @unpack model, M, RHS = cachePack
    @unpack H, iwG, iwE, f, diw = model.pr
    @unpack aRi1M, aIi1M, lRi1M, lIi1M, Dn, Dmt = model.id
    @unpack ai1M, li1M = model.id
    @unpack iterRtol, apopThres = model.pr

    ca = getAlpha(x, 1, model)
    cl = getLambda(x, 1, model)
    S = getS(cl, cl)
    
    Dist = getDistance(cl, cl, model)    
    Dist = (Dist + (apopThres+1.0)*Diagonal(ones(Float64, model.nd.nmultp1M))) .< apopThres
    iDist = weakly_connected_components(SimpleGraph(Dist))
    iDist = iDist[length.(iDist) .> 1]
    apopQ = length(iDist) != 0 ? true : false 

    @tullio  rho[i,j,n,m] := conj(ca[i,n])*ca[j,m]

    @tullio  G[i,j,n,m] := rho[i,j,n,m]*S[i,j]
    @tullio  P[i,j,n] := G[i,j,n,n]

    @tullio  rho[i,i,n,n] += 1e-7*Dmt[i]*Dn[n]

    @tullio  Pm[i,j,n] := rho[i,j,n,n]*S[i,j]

    @tullio  A[i,j] := iwG[k,h]*conj(cl[i,k,h])*cl[j,k,h]

    @tullio  C[i,j,n] := -iwE[n,h]*f[n,h]*(conj(cl[i,n,h])*Dmt[j] + cl[j,n,h]*Dmt[i])

    @tullio  RHS[ai1M[i,n]] =  ca[j,m]*S[i,j]*H[n,m]
    @tullio  RHS[ai1M[i,n]] += ca[j,n]*S[i,j]*A[i,j]
    @tullio  RHS[ai1M[i,n]] += ca[j,n]*S[i,j]*C[i,j,n]
    
    @tullio  RHS[li1M[i,k,h]] =  G[i,j,n,m]*cl[j,k,h]*H[n,m]
    @tullio  RHS[li1M[i,k,h]] += P[i,j,n]*cl[j,k,h]*A[i,j] 
    @tullio  RHS[li1M[i,k,h]] += P[i,j,n]*cl[j,k,h]*C[i,j,n]
    @tullio  RHS[li1M[i,k,h]] +=  P[i,j,n]*iwG[k,h]*cl[j,k,h]
    @tullio  RHS[li1M[i,k,h]] += -P[i,j,k]*iwE[k,h]*f[k,h]
    
    model.pr.anharmQ ? begin
        @tullio  D[i,j,n] := diw[n,h] * ( 1.0 + (conj(cl[i,n,h])*Dmt[j] + cl[j,n,h]*Dmt[i])^2 ) /4.0 
        
        @tullio  RHS[ai1M[i,n]] += ca[j,n]*S[i,j]*D[i,j,n]
        
        @tullio  RHS[li1M[i,k,h]] += P[i,j,n]*cl[j,k,h]*D[i,j,n]
        @tullio  RHS[li1M[i,k,h]] +=  P[i,j,k]*2.0*diw[k,h]*(conj(cl[i,k,h])*Dmt[j] + cl[j,k,h]*Dmt[i]) /4.0 
    end : nothing
    
    RHS .*= -im

    @tullio  iL[i,j,m,h] := conj(cl[i,m,h])*Dmt[j] - 0.5*conj(cl[j,m,h])*Dmt[i]
    @tullio  iLm[i,j,m,h] := iL[i,j,m,h] - 0.5*cl[j,m,h]*Dmt[i]
    @tullio  iLp[i,j,m,h] := iL[i,j,m,h] + 0.5*cl[j,m,h]*Dmt[i]

    M .= 0.0

    @tullio  M[aRi1M[i,n], aRi1M[j,n]] = real(      S[i,j] )
    @tullio  M[aRi1M[i,n], aIi1M[j,n]] = real(    im*S[i,j] )
    @tullio  M[aRi1M[i,n], lRi1M[j,m,f]] = real(   ca[j,n]*S[i,j]*iLm[i,j,m,f] )
    @tullio  M[aRi1M[i,n], lIi1M[j,m,f]] = real( im*ca[j,n]*S[i,j]*iLp[i,j,m,f] )

    @tullio  M[aIi1M[i,n], aRi1M[j,n]] = imag(      S[i,j] )
    @tullio  M[aIi1M[i,n], aIi1M[j,n]] = imag(    im*S[i,j] )
    @tullio  M[aIi1M[i,n], lRi1M[j,m,f]] = imag(   ca[j,n]*S[i,j]*iLm[i,j,m,f] )
    @tullio  M[aIi1M[i,n], lIi1M[j,m,f]] = imag( im*ca[j,n]*S[i,j]*iLp[i,j,m,f] )

    @tullio  M[lRi1M[i,k,h], aRi1M[j,n]] = real(      conj(ca[i,n])*S[i,j]*cl[j,k,h]  )
    @tullio  M[lRi1M[i,k,h], aIi1M[j,n]] = real(    im*conj(ca[i,n])*S[i,j]*cl[j,k,h]  )
    @tullio  M[lRi1M[i,k,h], lRi1M[j,k,h]] = real(    Pm[i,j,n] )
    @tullio  M[lRi1M[i,k,h], lIi1M[j,k,h]] = real(  im*Pm[i,j,n] )
    @tullio  M[lRi1M[i,k,h], lRi1M[j,m,f]] += real(   Pm[i,j,n]*cl[j,k,h]*iLm[i,j,m,f] )
    @tullio  M[lRi1M[i,k,h], lIi1M[j,m,f]] += real( im*Pm[i,j,n]*cl[j,k,h]*iLp[i,j,m,f] )

    @tullio  M[lIi1M[i,k,h], aRi1M[j,n]] = imag(      conj(ca[i,n])*S[i,j]*cl[j,k,h]  )
    @tullio  M[lIi1M[i,k,h], aIi1M[j,n]] = imag(    im*conj(ca[i,n])*S[i,j]*cl[j,k,h]  )
    @tullio  M[lIi1M[i,k,h], lRi1M[j,k,h]] = imag(    Pm[i,j,n] )
    @tullio  M[lIi1M[i,k,h], lIi1M[j,k,h]] = imag(  im*Pm[i,j,n] )
    @tullio  M[lIi1M[i,k,h], lRi1M[j,m,f]] += imag(   Pm[i,j,n]*cl[j,k,h]*iLm[i,j,m,f] )
    @tullio  M[lIi1M[i,k,h], lIi1M[j,m,f]] += imag( im*Pm[i,j,n]*cl[j,k,h]*iLp[i,j,m,f] )

    apopQ ? begin
        for key in iDist
            i = key[1]
            li1M_i = @view li1M[i,:,:]
            lRi1M_i = @view lRi1M[i,:,:]
            lIi1M_i = @view lIi1M[i,:,:]
            for k in 2:length(key)
                j = key[k]
                li1M_j = @view li1M[j,:,:]
                RHS[li1M_i] .+= RHS[li1M_j]
                RHS[li1M_j] .= -111.0
                lRi1M_j = @view lRi1M[j,:,:]
                lIi1M_j = @view lIi1M[j,:,:]
                M[lRi1M_i, :] .+= M[lRi1M_j, :]
                M[lIi1M_i, :] .+= M[lIi1M_j, :]
                M[:, lRi1M_i] .+= M[:, lRi1M_j]
                M[:, lIi1M_i] .+= M[:, lIi1M_j]
                M[lRi1M_j, :] .= -111.0
                M[lIi1M_j, :] .= -111.0
                M[:, lRi1M_j] .= -111.0
                M[:, lIi1M_j] .= -111.0
            end
        end

        iRHS = findall(RHS .== -111.0)
        setDiffRHS = setdiff(1:length(x), iRHS)
        RHS = @view RHS[setDiffRHS]

        iM = findall(M[1,:] .== -111.0)
        setDiffM = setdiff(1:size(M)[1], iM)
        M = @view M[setDiffM, setDiffM]

        dx .= ComplexF64(0.0, 0.0)
        dx[setDiffRHS] .= reinterpret(ComplexF64, gmres(M, reinterpret(Float64, RHS), verbose=false, reltol = iterRtol, restart = 200, Pl = lu(M) ))

        for key in iDist
            i = key[1]
            li1M_i = @view li1M[i,:,:]
            for k in 2:length(key)
                j = key[k]
                li1M_j = @view li1M[j,:,:]
                dx[li1M_j] .= dx[li1M_i]
            end
        end

    end : dx .= reinterpret(ComplexF64, gmres(M, reinterpret(Float64, RHS), verbose=false, reltol = iterRtol, restart = 200, Pl = lu(M) ))

    # println("1M: ", t)
    return nothing
end

function mD2_0M_ODE_RI_APO!(dx::Vector{ComplexF64}, x::Vector{ComplexF64}, cachePack::cachePackStructFloat64, t::Float64)

    @unpack model, M, RHS = cachePack
    @unpack iwG = model.pr
    @unpack aRi0M, aIi0M, lRi0M, lIi0M, Dn0M, Dmt = model.id
    @unpack ai0M, li0M = model.id
    @unpack iterRtol, apopThres = model.pr

    iwG = @view iwG[:]

    ca = getAlpha(x, 0, model)
    cl = getLambda(x, 0, model)
    S = getS(cl, cl)
    
    Dist = getDistance(cl, cl, model)
    Dist = (Dist + (apopThres+1.0)*Diagonal(ones(Float64, model.nd.nmultp0M))) .< apopThres
    iDist = weakly_connected_components(SimpleGraph(Dist))
    iDist = iDist[length.(iDist) .> 1]
    apopQ = length(iDist) != 0 ? true : false 

    @tullio  rho[i,j] := conj(ca[i,1])*ca[j,1]

    @tullio  G[i,j] := rho[i,j]*S[i,j]
    @tullio  P[i,j] := G[i,j]

    @tullio  rho[i,i] += 1e-7*Dmt[i]

    @tullio  Pm[i,j] := rho[i,j]*S[i,j]

    @tullio  A[i,j] := iwG[h]*conj(cl[i,1,h])*cl[j,1,h]
    
    @tullio  RHS[ai0M[i,1]]   = ca[j,1]*S[i,j]*A[i,j]
        
    @tullio  RHS[li0M[i,1,h]]  = P[i,j]*cl[j,1,h]*A[i,j]
    @tullio  RHS[li0M[i,1,h]] += P[i,j]*iwG[h]*cl[j,1,h]
    
    RHS .*= -im
    
    @tullio  iL[i,j,h]  := conj(cl[i,1,h])*Dmt[j] - 0.5*conj(cl[j,1,h])*Dmt[i]
    @tullio  iLm[i,j,h] := iL[i,j,h] - 0.5*cl[j,1,h]*Dmt[i]
    @tullio  iLp[i,j,h] := iL[i,j,h] + 0.5*cl[j,1,h]*Dmt[i]

    M .= 0.0

    @tullio  M[aRi0M[i,1], aRi0M[j,1]] = real(      S[i,j] )
    @tullio  M[aRi0M[i,1], aIi0M[j,1]] = real(    im*S[i,j] )
    @tullio  M[aRi0M[i,1], lRi0M[j,1,f]] = real(   ca[j,1]*S[i,j]*iLm[i,j,f] )
    @tullio  M[aRi0M[i,1], lIi0M[j,1,f]] = real( im*ca[j,1]*S[i,j]*iLp[i,j,f] )

    @tullio  M[aIi0M[i,1], aRi0M[j,1]] = imag(      S[i,j] )
    @tullio  M[aIi0M[i,1], aIi0M[j,1]] = imag(    im*S[i,j] )
    @tullio  M[aIi0M[i,1], lRi0M[j,1,f]] = imag(   ca[j,1]*S[i,j]*iLm[i,j,f] )
    @tullio  M[aIi0M[i,1], lIi0M[j,1,f]] = imag( im*ca[j,1]*S[i,j]*iLp[i,j,f] )

    @tullio  M[lRi0M[i,1,h], aRi0M[j,1]] = real(      conj(ca[i,1])*S[i,j]*cl[j,1,h]  )
    @tullio  M[lRi0M[i,1,h], aIi0M[j,1]] = real(    im*conj(ca[i,1])*S[i,j]*cl[j,1,h]  )
    @tullio  M[lRi0M[i,1,h], lRi0M[j,1,h]] = real(    Pm[i,j] )
    @tullio  M[lRi0M[i,1,h], lIi0M[j,1,h]] = real(  im*Pm[i,j] )
    @tullio  M[lRi0M[i,1,h], lRi0M[j,1,f]] += real(   Pm[i,j]*cl[j,1,h]*iLm[i,j,f] )
    @tullio  M[lRi0M[i,1,h], lIi0M[j,1,f]] += real( im*Pm[i,j]*cl[j,1,h]*iLp[i,j,f] )

    @tullio  M[lIi0M[i,1,h], aRi0M[j,1]] = imag(      conj(ca[i,1])*S[i,j]*cl[j,1,h]  )
    @tullio  M[lIi0M[i,1,h], aIi0M[j,1]] = imag(    im*conj(ca[i,1])*S[i,j]*cl[j,1,h]  )
    @tullio  M[lIi0M[i,1,h], lRi0M[j,1,h]] = imag(    Pm[i,j] )
    @tullio  M[lIi0M[i,1,h], lIi0M[j,1,h]] = imag(  im*Pm[i,j] )
    @tullio  M[lIi0M[i,1,h], lRi0M[j,1,f]] += imag(   Pm[i,j]*cl[j,1,h]*iLm[i,j,f] )
    @tullio  M[lIi0M[i,1,h], lIi0M[j,1,f]] += imag( im*Pm[i,j]*cl[j,1,h]*iLp[i,j,f] )

    apopQ ? begin 
        for key in iDist
            i = key[1]
            li0M_i = @view li0M[i,:,:]
            lRi0M_i = @view lRi0M[i,:,:]
            lIi0M_i = @view lIi0M[i,:,:]
            for k in 2:length(key)
                j = key[k]
                li0M_j = @view li0M[j,:,:]
                RHS[li0M_i] .+= RHS[li0M_j]
                RHS[li0M_j] .= -111.0
                lRi0M_j = @view lRi0M[j,:,:]
                lIi0M_j = @view lIi0M[j,:,:]
                M[lRi0M_i, :] .+= M[lRi0M_j, :]
                M[lIi0M_i, :] .+= M[lIi0M_j, :]
                M[:, lRi0M_i] .+= M[:, lRi0M_j]
                M[:, lIi0M_i] .+= M[:, lIi0M_j]
                M[lRi0M_j, :] .= -111.0
                M[lIi0M_j, :] .= -111.0
                M[:, lRi0M_j] .= -111.0
                M[:, lIi0M_j] .= -111.0
            end
        end

        iRHS = findall(RHS .== -111.0)
        setDiffRHS = setdiff(1:length(x), iRHS)
        RHS = @view RHS[setDiffRHS]

        iM = findall(M[1,:] .== -111.0)
        setDiffM = setdiff(1:size(M)[1], iM)
        M = @view M[setDiffM, setDiffM]

        dx .= ComplexF64(0.0, 0.0)
        dx[setDiffRHS] .= reinterpret(ComplexF64, gmres(M, reinterpret(Float64, RHS), verbose=false, reltol = iterRtol, restart = 200, Pl = lu(M) ))
    
        for key in iDist
            i = key[1]
            li0M_i = @view li0M[i,:,:]
            for k in 2:length(key)
                j = key[k]
                li0M_j = @view li0M[j,:,:]
                dx[li0M_j] .= dx[li0M_i]
            end
        end
    
    end : dx .= reinterpret(ComplexF64, gmres(M, reinterpret(Float64, RHS), verbose=false, reltol = iterRtol, restart = 200, Pl = lu(M) ))

    # println("0M: ", t)
    return nothing
end
