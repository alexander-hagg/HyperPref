clear;
%% Configuration
addpath(genpath(pwd))                           % Set path to all modules
DOF = 16;                                       % Degrees of freedom, Catmull-Rom spline domain
d = domain(DOF);                                % Domain configuration
FITNESSFUNCTION = 'bmpSymmetry'; rmpath(genpath('domain/catmullRom/fitnessFunctions')); addpath(genpath(['domain/catmullRom/fitnessFunctions/' FITNESSFUNCTION]));
numInitSamples = 512;
baseFilename = ['catmullRom_IIIb'];

latentDOFs = [16];

ALGORITHM = 'voronoi'; rmpath('QD/mapelites'); rmpath('QD/voronoi'); addpath(['QD/' ALGORITHM]);
load('randomQDmap.mat', 'map')
%%
observations = map{1}.genes;

%sobSequence = scramble(sobolset(d.dof,'Skip',1e3),'MatousekAffineOwen');  sobPoint = (1-1)*numInitSamples+1;
%observations = range(d.ranges').*sobSequence(sobPoint:(sobPoint+numInitSamples)-1,:)+d.ranges(:,1)';
%[~,phenotypes] = fitfun(observations,d);
        
clear model losses
% latentDim  numFilters filterSize stride
numFilterss = [16 32 64]
numFilterss = [64]
filterSizes = [4 8 16]
for i = 1:length(numFilterss)
    for j = 1:length(filterSizes)
        numFilters = numFilterss(i)
        filterSize = filterSizes(j)
        % Run POEM on parameter space with latent space niching
        m = cfgLatentModel('data/workdir',d.resolution,latentDOFs(1),numFilters,filterSize);                    % VAE configuration
        %decoderLG = m.decoderLG
        %encoderLG = m.encoderLG
        %%
        model{i,j} = trainFeatures(phenotypes,m);
        losses(i,j,:) = model{i,j}.statistics.reconstructionLoss(model{i,j}.statistics.reconstructionLoss > 0);
    end
end


