function getGSB_NR_threaded(x0::Vector{ComplexF64}, eVec::Vector{Float64}, tProp::Vector{Float64}, tW::Float64, model::modelStruct)
    
    tM = tProp[end]
    tNsteps = length(tProp)
    k1 = transitionDipoleOperator(eVec, +1)
    k2 = transitionDipoleOperator(eVec, -1)
    k3 = transitionDipoleOperator(eVec, +1)
    kI = transitionDipoleOperator(eVec, -1, "L")
    
    GSB = Array{ComplexF64}(undef, tNsteps, tNsteps)

    Rt3 = createState(0, x0, 0.0, 1)
    propagate!(Rt3, tM+tW+tM, model)

    Lt1 = getNewState(k1, createState(0, x0, 0.0, 1), model)
    propagate!(Lt1, tM, model)

    Lt3 = Array{stateStruct}(undef, tNsteps)

    Threads.@threads for i in 1:tNsteps
        t1 = tProp[i]
        Lt3[i] = getNewState(k2, createStateAt(t1, Lt1, model), model)
        propagate!(Lt3[i], t1+tW, model)
        Lt3[i] = getNewState(k3, Lt3[i], model)
        propagate!(Lt3[i], t1+tW+tM, model)
    end

    Threads.@threads for j in 1:tNsteps
        for i in 1:tNsteps
            t1 = tProp[i]
            t3 = tProp[j]
            R = createStateAt(t1+tW+t3, Rt3, model)
            L = createStateAt(t1+tW+t3, Lt3[i], model)
            GSB[i,j] = expValueAt(t1+tW+t3, L, kI, R, model)
        end
    end

    return GSB
end

function getESE_NR_threaded(x0::Vector{ComplexF64}, eVec::Vector{Float64}, tProp::Vector{Float64}, tW::Float64, model::modelStruct)
    
    tM = tProp[end]
    tNsteps = length(tProp)
    k1 = transitionDipoleOperator(eVec, +1)
    k2 = transitionDipoleOperator(eVec, +1)
    k3 = transitionDipoleOperator(eVec, -1)
    kI = transitionDipoleOperator(eVec, -1, "L")

    ESE = Array{ComplexF64}(undef, tNsteps, tNsteps)

    Rt1 = createState(0, x0, 0.0, 1)
    propagate!(Rt1, tM, model)

    Lt3 = getNewState(k1, createState(0, x0, 0.0, 1), model)
    propagate!(Lt3, tM+tW+tM, model)

    Rt3 = Array{stateStruct}(undef, tNsteps)

    Threads.@threads for i in 1:tNsteps
        t1 = tProp[i]
        Rt3[i] = getNewState(k2, createStateAt(t1, Rt1, model), model)
        propagate!(Rt3[i],t1+tW, model)
        Rt3[i] = getNewState(k3, Rt3[i], model)
        propagate!(Rt3[i], t1+tW+tM, model)
    end

    Threads.@threads for j in 1:tNsteps
        for i in 1:tNsteps
            t1 = tProp[i]
            t3 = tProp[j]
            R = createStateAt(t1+tW+t3, Rt3[i], model)
            L = createStateAt(t1+tW+t3, Lt3, model)
            ESE[i,j] = expValueAt(t1+tW+t3, L, kI, R, model)
        end
    end

    return ESE
end

function getESA_NR_threaded(x0::Vector{ComplexF64}, eVec::Vector{Float64}, tProp::Vector{Float64}, tW::Float64, model::modelStruct)

    tM = tProp[end]
    tNsteps = length(tProp)
    k1 = transitionDipoleOperator(eVec, +1)
    k2 = transitionDipoleOperator(eVec, +1)
    k3 = transitionDipoleOperator(eVec, +1)
    kI = transitionDipoleOperator(eVec, -1, "L")

    ESA = Array{ComplexF64}(undef, tNsteps, tNsteps)

    Rt1 = createState(0, x0, 0.0, 1)
    propagate!(Rt1, tM, model)

    Lt2 = getNewState(k1, createState(0, x0, 0.0, 1), model)
    propagate!(Lt2, tM+tW, model)

    Rt3 = Array{stateStruct}(undef, tNsteps)
    Lt3 = Array{stateStruct}(undef, tNsteps)

    Threads.@threads for i in 1:tNsteps
        t1 = tProp[i]
        Rt3[i] = getNewState(k2, createStateAt(t1, Rt1, model), model)
        propagate!(Rt3[i], t1+tW+tM, model)
    end
    
    Threads.@threads for i in 1:tNsteps
        t1 = tProp[i]
        Lt3[i] = getNewState(k3, createStateAt(t1+tW, Lt2, model), model)
        propagate!(Lt3[i], t1+tW+tM, model)
    end

    Threads.@threads for j in 1:tNsteps
        for i in 1:tNsteps
            t1 = tProp[i]
            t3 = tProp[j]
            R = createStateAt(t1+tW+t3, Rt3[i], model)
            L = createStateAt(t1+tW+t3, Lt3[i], model)
            ESA[i,j] = -1.0*expValueAt(t1+tW+t3, L, kI, R, model)
        end
    end

    return ESA
end

function getESE_R_threaded(x0::Vector{ComplexF64}, eVec::Vector{Float64}, tProp::Vector{Float64}, tW::Float64, model::modelStruct)

    tM = tProp[end]
    tNsteps = length(tProp)
    k1 = transitionDipoleOperator(eVec, +1)
    k2 = transitionDipoleOperator(eVec, +1)
    k3 = transitionDipoleOperator(eVec, -1)
    kI = transitionDipoleOperator(eVec, -1, "L")

    ESE = Array{ComplexF64}(undef, tNsteps, tNsteps)

    Rt2 = getNewState(k1, createState(0, x0, 0.0, 1), model)
    propagate!(Rt2, tM+tW, model)

    Lt1 = createState(0, x0, 0.0, 1)
    propagate!(Lt1, tM, model)

    Rt3 = Array{stateStruct}(undef, tNsteps)
    Lt3 = Array{stateStruct}(undef, tNsteps)

    Threads.@threads for i in 1:tNsteps
        t1 = tProp[i]
        Rt3[i] = getNewState(k3, createStateAt(t1+tW, Rt2, model), model)
        propagate!(Rt3[i], t1+tW+tM, model)
    end

    Threads.@threads for i in 1:tNsteps
        t1 = tProp[i]
        Lt3[i] = getNewState(k2, createStateAt(t1, Lt1, model), model)
        propagate!(Lt3[i], t1+tW+tM, model)
    end

    Threads.@threads for j in 1:tNsteps
        for i in 1:tNsteps
            t1 = tProp[i]
            t3 = tProp[j]
            R = createStateAt(t1+tW+t3, Rt3[i], model)
            L = createStateAt(t1+tW+t3, Lt3[i], model)
            ESE[i,j] = expValueAt(t1+tW+t3, L, kI, R, model)
        end
    end

    return ESE
end

function getGSB_R_threaded(x0::Vector{ComplexF64}, eVec::Vector{Float64}, tProp::Vector{Float64}, tW::Float64, model::modelStruct)

    tM = tProp[end]
    tNsteps = length(tProp)
    k1 = transitionDipoleOperator(eVec, +1)
    k2 = transitionDipoleOperator(eVec, -1)
    k3 = transitionDipoleOperator(eVec, +1)
    kI = transitionDipoleOperator(eVec, -1, "L")
    GSB = Array{ComplexF64}(undef, tNsteps, tNsteps)

    Rt1 = getNewState(k1, createState(0, x0, 0.0, 1), model)
    propagate!(Rt1, tM, model)

    Lt2 = createState(0, x0, 0.0, 1)
    propagate!(Lt2, tM+tW, model)

    Rt3 = Array{stateStruct}(undef, tNsteps)
    Lt3 = Array{stateStruct}(undef, tNsteps)

    Threads.@threads for i in 1:tNsteps
        t1 = tProp[i]
        Rt3[i] = getNewState(k2, createStateAt(t1, Rt1, model), model)
        propagate!(Rt3[i], t1+tW+tM, model)
    end 

    Threads.@threads for i in 1:tNsteps
        t1 = tProp[i]
        Lt3[i] = getNewState(k3, createStateAt(t1+tW, Lt2, model), model)
        propagate!(Lt3[i], t1+tW+tM, model)
    end

    Threads.@threads for j in 1:tNsteps
        for i in 1:tNsteps
            t1 = tProp[i]
            t3 = tProp[j]
            R = createStateAt(t1+tW+t3, Rt3[i], model)
            L = createStateAt(t1+tW+t3, Lt3[i], model)
            GSB[i,j] = expValueAt(t1+tW+t3, L, kI, R, model)
        end
    end

    return GSB
end

function getESA_R_threaded(x0::Vector{ComplexF64}, eVec::Vector{Float64}, tProp::Vector{Float64}, tW::Float64, model::modelStruct)

    tM = tProp[end]
    tNsteps = length(tProp)
    k1 = transitionDipoleOperator(eVec, +1)
    k2 = transitionDipoleOperator(eVec, +1)
    k3 = transitionDipoleOperator(eVec, +1)
    kI = transitionDipoleOperator(eVec, -1, "L")

    ESA = Array{ComplexF64}(undef, tNsteps, tNsteps)

    Rt3 = getNewState(k1, createState(0, x0, 0.0, 1), model)
    propagate!(Rt3, tM+tW+tM, model)

    Lt1 = createState(0, x0, 0.0, 1)
    propagate!(Lt1, tM, model)

    Lt3 = Array{stateStruct}(undef, tNsteps)

    Threads.@threads for i in 1:tNsteps
        t1 = tProp[i]
        Lt3[i] = getNewState(k2, createStateAt(t1, Lt1, model), model)
        propagate!(Lt3[i], t1+tW, model)
        Lt3[i] = getNewState(k3, Lt3[i], model)
        propagate!(Lt3[i], t1+tW+tM, model)
    end

    Threads.@threads for j in 1:tNsteps
        for i in 1:tNsteps
            t1 = tProp[i]
            t3 = tProp[j]
            R = createStateAt(t1+tW+t3, Rt3, model)
            L = createStateAt(t1+tW+t3, Lt3[i], model)
            ESA[i,j] = -1.0*expValueAt(t1+tW+t3, L, kI, R, model)
        end
    end

    return ESA
end
