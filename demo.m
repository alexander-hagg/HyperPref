clear;clc;
%% Configuration
addpath('.'); % Set path to all modules
DOF = 16;DOMAIN = 'catmullRom';                 % Degrees of freedom, Catmull-Rom spline domain
d = domain(DOF);                                % Domain configuration
p = defaultParamSet;                            % Base Quality Diversity (QD) configuration (MAP-Elites)
m = cfgLatentModel('data/workdir',d.resolution);% VAE configuration

poemCfg = poemParamSet(p,m);                    % Configure POEM ("Phenotypic niching based Optimization by Evolving a Manifold")
poemCfg.categorize = @(geno,pheno,p,d) predictFeatures(pheno,p.model);  % Anonymous function ptr to phenotypic categorization function (= VAE)

%% Initialize Experiment
% Initialize solution set using space filling Sobol sequence in genetic
% space
sobSequence = scramble(sobolset(d.dof,'Skip',1e3),'MatousekAffineOwen');  sobPoint = 1;
initSamples = range(d.ranges').*sobSequence(sobPoint:(sobPoint+poemCfg.map.numInitSamples)-1,:)+d.ranges(:,1)';
[fitness,polygons] = fitfun(initSamples,d);

%% Run POEM's first iteration
[map{1}, config{1}, stats{1}] = poem(initSamples,polygons,fitness,poemCfg,d,2);
save([DOMAIN 'step1.mat']);
disp('HyperPref Step 1 Done');

%% Reload and extract results of first iteration and select IDs of shapes
load([DOMAIN 'step1.mat']);
[genes,fitness,features,bins] = extractMap(map{1});

selectionIDs = [20, 25]; % Selected shapes (IDs in QD archive, see figure)
d.userModel = stats{1}.models{end}; % Save user model to use as constraint model

phenotypes = d.getPhenotype(genes);
features = predictFeatures(phenotypes,d.userModel);
d.selectedShapes = features(selectionIDs,:);
d.deselectedShapes = features; d.deselectedShapes(selectionIDs,:) = [];

% Visualization
cmap = [0 0 0; 0 0 1]; colors = repmat(cmap(1,:),size(genes,1),1); colors(selectionIDs,:) = repmat(cmap(2,:),numel(selectionIDs),1);
showPhenotype(genes,d, [], bins,colors);

%% Perturb selected shapes
newSamples = genes(selectionIDs,:);
nNewPerSelected = ceil(poemCfg.map.numInitSamples./length(selectionIDs));
for i=1:length(selectionIDs)
    newSampleMutations = 0.1 * randn(nNewPerSelected,d.dof);
    newSamples = [newSamples; genes(selectionIDs(i),:) + newSampleMutations];
end

showPhenotype(newSamples,d,[]);title('Injected Shapes');

%% Run POEM's second iteration based on the user selection
[newSamplesfitness,newSamplespolygons] = fitfun(newSamples,d); % Recalculate fitness! (User selection influences fitness values)
[map{2}, config{2}, stats{2}] = poem(newSamples,newSamplespolygons,newSamplesfitness,poemCfg,d,2);

close all; clear figs; save([DOMAIN 'step2.mat']);
disp('HyperPref Step 2 Done');


%% Reload and extract results of second iteration and visualize
load([DOMAIN 'step2.mat']);
[genes,fitness,features,bins] = extractMap(map{2});

% Visualization
showPhenotype(genes,d); title('2nd Iteration Result after Selection'); axis equal;