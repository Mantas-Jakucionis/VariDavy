@with_kw mutable struct Example_Model

      atol = 1e-05 # absolute tolerance for the "propagationMethod" 
      rtol = 1e-08 # relative tolerance for the "propagationMethod"
      
      iterRtol = 1e-14 # relative tolerance for the GMRES() solver

      apopThres = 0.1 # apoptosis threshold
      d2Thres = 0.1 # threshold for mD2 -> D2 approximation 

      singleMultipleLES = false # whther to optimize for the lowest energy state in terms of a single multiple
      ntrialsLES = 1 # number of trials for optimization for the lowest energy state 

      multpDistr = "polygon" # polygon/grid; function to place unpopulated multiples  
      multpPerLayer = 4 # for polygon; number of multiples per polygon layer
      multpDistance = 1 # for polygon; distance between polygon layers
      
      propagateSimply = 1 # uses D2 instead of mD2 where applicable
      propagationMethod = VCABM() # method to propagate equations of motion

      thermalization = 0 # whether to use thermalization
      thermTempOffset = 0.0 # secondary bath temperature offset with respect to "temp" 
      thermMethod = "simple" # use simple
      thermType = "mode" # mode/bath/global; consider probabilities to thermalize either each mode, each bath or all modes (global) 
      thermStep = 10 # thermalization step size
      thermRate = 0.001 # thermalization scattering rate

      trfThermTime = 0.0 # only for getTrfResp(; trfLES="prop-therm"); equilibration with thermalization time

      particleType = "boson" # type of electronic state particles 

      nmultp = 1 # number of multiples
      nsite = 3 # number of sites
      nibath = nsite # number of baths; now each site has a local bath 

      H = zeros(Float64, nsite, nsite) # single-excitation Hamiltonian matrix {nsite, nsite}

      K = zeros(Float64, nsite, nsite) # double-excitation Hamiltonian matrix {nsite, nsite}
      
      dipGE = nothing # transition dipole vectors {nsite, 3}
      
      rsite = nothing # reorganization energy of each site {nsite}

      iwG = nothing # frequencies of intramolecular vibrations in the ground state {nsite, Q}
      
      iwE = nothing # frequencies of intramolecular vibrations in the excited state {nsite, Q}

      id = nothing # QHO displacements of intramolecular vibrations in the excited state {nsite, Q}

      staticDisorderSigma = 0.0 # standard deviation of Gaussian static energy disorder for site excitation energies
      
      extFieldDistribution = "uniform" # static/uniform; direction of external electric field vector. sampled uniformly in space if "uniform"
      extEstatic = [] # vector of external electric field vector if "extFieldDistribution" is static 

      bathSpectralDensity = Drude # spectral density function (SPD) of bath
      bathSpdParameters = [100] # necessary parameters of the selected SPD
      tmin = 0.0 # initital propagation time
      tmax = 2000.0 # final propagation time
      dt   = 1.0 # time step size to sample interval [tmin, tmax]

      temp = 300.0 # initial bath temperature
      ntraj = 1 # number of trajectories

      wmin = 0.1 # smallest frequency for sampling bath SPD
      wmax = 200 # largest frequency for sampling bath SPD; if bath modes should be excluded, set to 0
      dw = 5 # frequency step size to sample bath SPD
      dwp = 1.00 # scaling factor of bath mode frequencies in the excited state compared to the ground state

      twaiting2D = 0.0 # length of t2 (waiting) time interval for 2DES Feynman diagrams
      tmin2D     = 0.0 # the initial time value of t1 and t3 time intervals for 2DES Feynman diagrams
      tmax2D     = 1000.0 # the final time value of t1 and t3 time intervals for 2DES Feynman diagrams
      dt2D       = 15.0 # time step size to sample interval [tmin2D, tmax2D]
end