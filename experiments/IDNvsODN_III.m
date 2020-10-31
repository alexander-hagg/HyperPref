clear;
%% Configuration
addpath(genpath(pwd))                           % Set path to all modules
DOF = 16;DOMAIN = 'catmullRom';                 % Degrees of freedom, Catmull-Rom spline domain
d = domain(DOF);                                % Domain configuration
latentDomain = d; latentDomain.dof = 2; latentDomain.ranges = [-3 3; -3 3];

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


%% Run POEM on latent space with latent space niching
[initLatentSamples ,xPred,xTrue]= getPrediction(phenotypes,results{1}.models(1));
latentDomain.getPhenotype = @(latentCoords)sampleVAE(latentCoords,results{1}.models(1).decoderNet);
[map{2}, config{2}, results{2}] = poem(initLatentSamples,pm,latentDomain,results{1}.models(1));
save([DOMAIN '_step2.mat']);

% TODO does not work with latent dimensions other than 2

%% Visualize
fig(1) = figure(1); viewMap(map{1},d)
fig(2) = figure(2); viewMap(map{2},latentDomain)

%%
genes = reshape(map{1}.genes,[],d.dof); genes = genes(all(~isnan(genes)'),:);
placement = reshape(map{1}.features,[],2); placement = placement(all(~isnan(placement)'),:);
showPhenotypeBMP(genes,d,[],placement)

genes = reshape(map{2}.genes,[],latentDomain.dof); genes = genes(all(~isnan(genes)'),:);
placement = reshape(map{2}.features,[],2); placement = placement(all(~isnan(placement)'),:);
input = []; input(1,1,:,:) = genes'; input = dlarray(input,'SSCB');
modelOutput = sigmoid(predict(results{1}.models(1).decoderNet, input));
modelOutput = gather(extractdata(modelOutput));    %reproduced
clear bitmaps;
for i=1:size(modelOutput,4)
    bitmaps{i} = squeeze(modelOutput(:,:,1,i));
end
showPhenotypeBMP(bitmaps,d,[],placement)

%% Run POEM on parameter space with latent space niching
initSamples2 = reshape(map{1}.genes,[],d.dof);
initSamples2 = initSamples2(all(~isnan(initSamples2)'),:);
[map{3}, config{3}, results{3}] = poem(initSamples2,pm,d);
save([DOMAIN '_step3.mat']);
disp('Finished POEM on parameter space');


%% Run POEM on latent space with latent space niching
initLatentSamples2 = predictFeatures(phenotypes,results{3}.models(1));
latentDomain.getPhenotype = @(latentCoords)sampleVAE(latentCoords,results{3}.models(1).decoderNet);
[map{4}, config{4}, results{4}] = poem(initLatentSamples,pm,latentDomain,results{3}.models(1));
save([DOMAIN '_step4.mat']);

%% Visualize
fig(5) = figure(5); viewMap(map{3},d)
fig(6) = figure(6); viewMap(map{4},latentDomain)

%%
genes = reshape(map{3}.genes,[],d.dof); showPhenotypeBMP(genes,d)
genes = reshape(map{4}.genes,[],latentDomain.dof); showPhenotypeBMP(genes,d)


%%
% save_figures(fig, '.', 'IDNODNIII', 12, [5 5])





