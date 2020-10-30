clear;
%% Configuration
addpath(genpath(pwd))                           % Set path to all modules
DOF = 16;DOMAIN = 'catmullRom';                 % Degrees of freedom, Catmull-Rom spline domain
d = domain(DOF);                                % Domain configuration
p = defaultParamSet;                            % Base Quality Diversity (QD) configuration (MAP-Elites)
m = cfgLatentModel('data/workdir',d.resolution);% VAE configuration
pm = poemParamSet(p,m);                         % Configure POEM ("Phenotypic niching based Optimization by Evolving a Manifold")
pm.categorize = @(geno,pheno,p,d) predictFeatures(pheno,p.model);  % Anonymous function ptr to phenotypic categorization function (= VAE)
FITNESSFUNCTION = 'bmpSymmetry'; rmpath(genpath('domain/catmullRom/fitnessFunctions')); addpath(genpath(['domain/catmullRom/fitnessFunctions/' FITNESSFUNCTION]));

%% Initialize Experiment
% Initialize solution set using space filling Sobol sequence in genetic space
sobSequence = scramble(sobolset(d.dof,'Skip',1e3),'MatousekAffineOwen');  sobPoint = 1;
initSamples = range(d.ranges').*sobSequence(sobPoint:(sobPoint+pm.map.numInitSamples)-1,:)+d.ranges(:,1)';
[fitness,phenotypes,rawFitness] = fitfun(initSamples,d);

%% Run POEM on parameter space with latent space niching
[map{1}, config{1}, results{1}] = poem(initSamples,pm,d);
save([DOMAIN '_step1.mat']);
disp('Finished POEM on parameter space');

%% Visualize
%viewMap(map{1},d)
genes = reshape(map{1}.genes,[],d.dof);
showPhenotypeBMP(genes,d)

[fit,~,raw] = fitfun(genes(all(~isnan(genes)'),:),d)
%% Run POEM on latent space with latent space niching
initLatentSamples = predictFeatures(phenotypes,results{1}.models(1));
latentDomain = d;
latentDomain.dof = 2;
latentDomain.ranges = [-3 3; -3 3];

latentDomain.getPhenotype = @(latentCoords)sampleVAE(latentCoords,results{1}.models(1).decoderNet);
[map{2}, config{2}, results{2}] = poem(initLatentSamples,pm,latentDomain,results{1}.models(1));

% TODO does not work with latent dimensions other than 2

%%
%viewMap(map{2},latentDomain)
genes = reshape(map{2}.genes,[],latentDomain.dof);
showPhenotypeBMP(genes,d)


%%
% save_figures(fig, '.', 'IDNODNIII', 12, [5 5])





