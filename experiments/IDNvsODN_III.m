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


%% Run POEM on parameter space with latent space niching
initSamples2 = reshape(map{1}.genes,[],d.dof);
initSamples2 = initSamples2(all(~isnan(initSamples2)'),:);
[map{3}, config{3}, results{3}] = poem(initSamples2,pm,d);
save([DOMAIN '_step3.mat']);
disp('Finished POEM on parameter space');


%% Run POEM on latent space with latent space niching
initLatentSamples2 = getPrediction(phenotypes,results{3}.models(1));
latentDomain.getPhenotype = @(latentCoords)sampleVAE(latentCoords,results{3}.models(1).decoderNet);
[map{4}, config{4}, results{4}] = poem(initLatentSamples2,pm,latentDomain,results{3}.models(1));
save([DOMAIN '_step4.mat']);

%% Run POEM on parameter space with latent space niching
initSamples3 = reshape(map{3}.genes,[],d.dof);
initSamples3 = initSamples3(all(~isnan(initSamples3)'),:);
[map{5}, config{5}, results{5}] = poem(initSamples3,pm,d);
save([DOMAIN '_step5.mat']);
disp('Finished POEM on parameter space');

%% Run POEM on latent space with latent space niching
initLatentSamples3 = getPrediction(phenotypes,results{5}.models(1));
latentDomain.getPhenotype = @(latentCoords)sampleVAE(latentCoords,results{5}.models(1).decoderNet);
[map{6}, config{6}, results{6}] = poem(initLatentSamples3,pm,latentDomain,results{5}.models(1));
save([DOMAIN '_step6.mat']);

%% Visualize
for i=1:3
    % Show feature map - search: parameter space
    fig((i-1)*4+1) = figure((i-1)*4+1); viewMap(map{(i-1)*2+1},d)
    
    % Show feature map - search: latent space
    fig((i-1)*4+2) = figure((i-1)*4+2); viewMap(map{(i-1)*2+2},latentDomain)
    
    % Show shapes - search: parameter space
    genes = reshape(map{(i-1)*2+1}.genes,[],d.dof); genes = genes(all(~isnan(genes)'),:);
    placement = reshape(map{(i-1)*2+1}.features,[],2); placement = placement(all(~isnan(placement)'),:);
    fig((i-1)*4+3) = figure((i-1)*4+3);
    showPhenotypeBMP(genes,d,fig((i-1)*4+3),placement);
    
    % Show shapes - search: latent space
    genes = reshape(map{(i-1)*2+2}.genes,[],latentDomain.dof); genes = genes(all(~isnan(genes)'),:);
    placement = reshape(map{(i-1)*2+2}.features,[],2); placement = placement(all(~isnan(placement)'),:);
    input = []; input(1,1,:,:) = genes'; input = dlarray(input,'SSCB');
    modelOutput = sigmoid(predict(results{(i-1)*2+1}.models(1).decoderNet, input));
    modelOutput = gather(extractdata(modelOutput));    %reproduced
    clear bitmaps;
    for j=1:size(modelOutput,4)
        bitmaps{j} = squeeze(modelOutput(:,:,1,j));
    end
    fig((i-1)*4+4) = figure((i-1)*4+4);
    showPhenotypeBMP(bitmaps,d,fig((i-1)*4+4),placement);
end


BMIN = 0.5; BMAX = 1; YMAX = 80;
fig(end+1) = figure;
for i=1:6
subplot(3,2,i);
histogram(map{i}.fitness(:),20,'BinLimits',[BMIN,BMAX],'Normalization','count')
ax = gca;ax.YLim = [0 YMAX];grid on;title('fitness');
end

%%
save_figures(fig, '.', 'IDNODNIII', 12, [5 5])


%%
genes = reshape(map{1}.genes,[],d.dof); genes = genes(all(~isnan(genes)'),:);
[~,flatbitmaps] = d.getPhenotype(genes);
[score(1),distMetric] = metricPD(flatbitmaps, 'hamming');

genes = reshape(map{2}.genes,[],latentDomain.dof); genes = genes(all(~isnan(genes)'),:);
[bitmaps] = latentDomain.getPhenotype(genes);
flatbitmaps = [];
for i=1:length(bitmaps)
    flatbitmaps(i,:) = imbinarize(bitmaps{i}(:),0.9);
end
[score(2),distMetric] = metricPD(flatbitmaps, 'hamming');


genes = reshape(map{3}.genes,[],d.dof); genes = genes(all(~isnan(genes)'),:);
[~,flatbitmaps] = d.getPhenotype(genes);
[score(3),distMetric] = metricPD(flatbitmaps, 'hamming');

genes = reshape(map{4}.genes,[],latentDomain.dof); genes = genes(all(~isnan(genes)'),:);
[bitmaps] = latentDomain.getPhenotype(genes);
flatbitmaps = [];
for i=1:length(bitmaps)
    flatbitmaps(i,:) = imbinarize(bitmaps{i}(:),0.9);
end
[score(4),distMetric] = metricPD(flatbitmaps, 'hamming');

genes = reshape(map{5}.genes,[],d.dof); genes = genes(all(~isnan(genes)'),:);
[~,flatbitmaps] = d.getPhenotype(genes);
[score(5),distMetric] = metricPD(flatbitmaps, 'hamming');

genes = reshape(map{6}.genes,[],latentDomain.dof); genes = genes(all(~isnan(genes)'),:);
[bitmaps] = latentDomain.getPhenotype(genes);
flatbitmaps = [];
for i=1:length(bitmaps)
    flatbitmaps(i,:) = imbinarize(bitmaps{i}(:),0.9);
end
[score(6),distMetric] = metricPD(flatbitmaps, 'hamming');

score
