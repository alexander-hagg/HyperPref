function config = poemParamSet(mapDefaults,AEDefaults)
%POEMPARAMSET 

config.map                           = mapDefaults;
config.model                         = AEDefaults;
config.map.numInitSamples            = 64;

config.retryInvalid                  = true;
config.numIterations                 = 1;

% Visualization and data management
config.display.illu                 = false;
config.display.illuMod              = 1;
end

