function model = cfgLatentModel(workDir,resolution, varargin)
%CFGLATENTMODEL cfguration of Variational Autoencoder
%
% Author: Alexander Hagg
% Bonn-Rhein-Sieg University of Applied Sciences (HBRS)
% email: alexander.hagg@h-brs.de
% Nov 2019; Last revision: 13-Nov-2019
%
%------------- BEGIN CODE --------------

lossFunction                  = 'beta_annealing'; % OPTIONS: 'tcbeta' 'beta' 'beta_annealing'
disp(['VAE with loss function: ' lossFunction]);
rmpath(genpath('latentmodels/VAE/lossFunctions')); addpath(genpath(['latentmodels/VAE/lossFunctions/loss_' lossFunction]));
mkdir(workDir);
structure                           = 'convDefault2';
configuration.latentDim             = 16;
if nargin > 2
    configuration.latentDim = varargin{1};
    disp(['Latent dims: ' int2str(configuration.latentDim)]);
end

configuration.numFilters                    = 32;
configuration.trainPerc                     = 1.00;
configuration.numEpochs                     = 1000;
configuration.maxBatchSize                  = 128;
configuration.learnRate                     = 1e-3;
configuration.filterSize                    = 4;
if nargin > 3
    configuration.filterSize = varargin{2};
    disp(['configuration.filterSize: ' int2str(configuration.filterSize)]);
end

configuration.stride                        = 1;
if nargin > 4
    configuration.stride = varargin{3};
    disp(['configuration.stride: ' int2str(configuration.stride)]);
end

model                             = feval(structure,configuration.latentDim,resolution,configuration.numFilters,configuration.filterSize,configuration.stride);
model.predict                     = @(phenotypes,m) getLatent(phenotypes,m.model)';
model.uncertainty                 = @(genomes,latent,manifold,getPhenotypes) getReconstructionError(genomes,latent,manifold,getPhenotype);
model.train                       = @(phenotypes) trainLatentModel(model,getDataPoly(phenotypes,workDir,resolution,configuration.trainPerc),configuration.numEpochs,configuration.maxBatchSize,configuration.learnRate);
model.configuration               = configuration;

end

%------------- END CODE --------------
