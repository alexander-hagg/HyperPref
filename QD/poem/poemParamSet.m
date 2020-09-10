function config = poemParamSet(mapDefaults,AEDefaults)
%POEMPARAMSET 

config.map                           = mapDefaults;
config.model                         = AEDefaults;

config.retryInvalid                  = true;
config.numIterations                 = 2;
%config.numInitSamples                = 32;

% Visualization and data management
config.display.illu              = false;
config.display.illuMod           = 1;
end

