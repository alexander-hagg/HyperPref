clear;
%% Configuration
addpath(genpath(pwd))                           % Set path to all modules
DOF = 16;DOMAIN = 'catmullRom';                 % Degrees of freedom, Catmull-Rom spline domain
d = domain(DOF);                                % Domain configuration
p = defaultParamSet;                            % Base Quality Diversity (QD) configuration (MAP-Elites)
numLatentDims = 2;
m = cfgLatentModel('data/workdir',d.resolution, numLatentDims);% VAE configuration
pm = poemParamSet(p,m);                         % Configure POEM ("Phenotypic niching based Optimization by Evolving a Manifold")
pm.categorize = @(geno,pheno,p,d) predictFeatures(pheno,p.model);  % Anonymous function ptr to phenotypic categorization function (= VAE)
hypID = 1;
%% Initialize Experiment
% Initialize solution set using space filling Sobol sequence in genetic space
clear matchedParams;
fig = figure(1);ax=gca;
%matchedParams(1,:) = [1 1 1 1 1 1 1 1 zeros(1,8)];
matchedParams(1,:) = [1 0.3 1 0.3 1 0.3 1 0.3 zeros(1,8)];

fig(1) = figure(1); hold off; ax = gca;
showPhenotype(matchedParams,d,1.2,ax); axis([-2 2 -2 2]); axis equal;

numShapes = 10;
scaling = [0.2 1.0];
rotation = [0 0.45*pi];
%load('matchedParams.mat');

scaleArray = [scaling(1):(scaling(2)-scaling(1))/(numShapes-1):scaling(2)]';
rotationArray = [rotation(1):(rotation(2)-rotation(1))/(numShapes-1):rotation(2)]';

genomes = []; iter = 1;
for i=1:size(matchedParams,1)
    for j=1:length(scaleArray)
        for k=1:length(rotationArray)
            genomes(j,k,:) = [scaleArray(j).*matchedParams(i,1:DOF/2) rotationArray(k) + matchedParams(i,DOF/2+1:end)];
        end
    end
end

[fitness1,phenotypes{1}] = fitfun(reshape(genomes,[],d.dof),d);
fig(2) = figure(2); hold off; ax = gca;
allGenomes{1} = reshape(genomes,[],d.dof);
showPhenotype(allGenomes{1},d,1.2,ax); axis equal;

allGenomes{2} = reshape(genomes([1:5 8:10],[1:5 8:10],:),[],d.dof);
[fitness2,phenotypes{2}] = fitfun(allGenomes{2},d);
fig(3) = figure(3); hold off; ax = gca;
showPhenotype(allGenomes{2},d,1.2,ax); axis equal;

allGenomes{3} = reshape(genomes([1:8],[1:8],:),[],d.dof);
[fitness3,phenotypes{3}] = fitfun(allGenomes{3},d);
fig(4) = figure(4); hold off; ax = gca;
showPhenotype(allGenomes{3},d,1.2,ax); axis equal;





%% Train latent models

allModels{1} = trainFeatures(phenotypes{1},m);
allModels{2} = trainFeatures(phenotypes{2},m);
allModels{3} = trainFeatures(phenotypes{3},m);

save([DOMAIN '_step2.mat']);

%% show sampling of latent space of VAE trained on random inputs.
for i=1:3
    minFeatures = 1.5*min(features{i}(:));
    maxFeatures = 1.5*max(features{i}(:));
    nSamples = ceil(1*numShapes);

    x = minFeatures:(maxFeatures-minFeatures)/nSamples:maxFeatures; y = x;
    [X,Y] = ndgrid(x,y);
    varyCoords = [X(:),Y(:)]';

    clear input;
    input(1,1,:,:) = varyCoords;
    input = dlarray(input,'SSCB');
    decoderNet = allModels{i}.decoderNet;
    generatedImage = sigmoid(predict(decoderNet, input));
    generatedImage = gather(extractdata(generatedImage));   
    
    [~,flatPhenotypes] = getPhenotypeBoolean(phenotypes{i}, allModels{i}.encoderLG.Layers(1).InputSize(1));
    features{i} = getPrediction(flatPhenotypes,allModels{i});
    clear input;
    input(1,1,:,:) = features{i}';
    input = dlarray(input,'SSCB');
    generatedImage2 = sigmoid(predict(allModels{i}.decoderNet, input));
    generatedImage2 = gather(extractdata(generatedImage2));
    
    scale = d.resolution;
    bitmapCoords = 1 + (ceil(scale*varyCoords)-min(ceil(scale*varyCoords(:))));
    imgSize = [0 -min(bitmapCoords(:)) + max(bitmapCoords(:))+scale];
    clear img;
    img = zeros(range(imgSize),range(imgSize));
    imgStep2 = img;
    for jj=1:size(generatedImage,4)
        coords = bitmapCoords(:,jj)';
        img([coords(1):(coords(1)+scale-1)],[coords(2):(coords(2)+scale-1)]) = img([coords(1):(coords(1)+scale-1)],[coords(2):(coords(2)+scale-1)]) + squeeze(generatedImage(:,:,:,jj));
    end
    img = img > 0.9;
    trainCoords = 1 + (ceil(scale*features{i})-scale*min(ceil(varyCoords(:))));
    for jj=1:size(generatedImage2,4)
        coords = trainCoords(jj,:);
        imgStep2([coords(1):(coords(1)+scale-1)],[coords(2):(coords(2)+scale-1)]) = imgStep2([coords(1):(coords(1)+scale-1)],[coords(2):(coords(2)+scale-1)]) + squeeze(generatedImage2(:,:,:,jj));
    end
    imgStep2 = imgStep2 > 0.9;
    
    imgComplete = 0.5*img+imgStep2;
    %img = mapminmax(img,0,1);
    fig(99+i) = figure(99+i);hold off; ax=gca;
    imshow(imgComplete)
    colormap([1 1 1; 0 0 0; 0 0 1]);
    %colormap(flipud(gray));
    drawnow    
    
end

save_figures(fig, '.', 'IDNODN', 12, [5 5])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run POEM's first iteration
[map{hypID}, config{hypID}, results{hypID}] = poem(initSamples,polygons,fitness,pm,d,2);
save([DOMAIN '_step1.mat']);
disp('HyperPref Step 1 Done');

%% Reload and extract results of first iteration and select IDs of shapes
load([DOMAIN '_step1.mat']);
[genes,fitness,features,bins] = extractMap(results{1}.maps{1});
% Visualization
colors = [zeros(size(genes,1),1) fitness zeros(size(genes,1),1)];            
showPhenotype(genes, d, 1.1, [], bins, colors); title('1st Iteration Result');
fig(1) = gcf;




%% in-distribution
[phenotypes,ctlPts] = d.getPhenotype(genes);
modelPost1 = trainFeatures(phenotypes,pm.model);
%%
clear input;

rr = randn(m.configuration.latentDim-2,1); % set 14 out of the 16 coordinates
x = -3:0.5:3; y = x;
[X,Y] = ndgrid(x,y);
varyCoords = [X(:),Y(:)]';
rawInput = [varyCoords;repmat(rr,1,size(varyCoords,2))];
randID = randi(m.configuration.latentDim,1,1);
rawInput([randID 1],:)=rawInput([1 randID],:);
randID = randi(m.configuration.latentDim,1,1);
rawInput([randID 2],:)=rawInput([2 randID],:);

input(1,1,:,:) = rawInput;
input = dlarray(input,'SSCB');
decoderNet = modelPost1.decoderNet;
generatedImage = sigmoid(predict(decoderNet, input));
generatedImage = extractdata(generatedImage);
fig(3) = figure;
imshow(imtile(generatedImage, "ThumbnailSize", [d.resolution,d.resolution]))
title("Generated by VAE by varying two dimensions, after retraining, iteration 1");

save_figures(fig, '.', 'hyperprefVAE', 12, [5 5])

%% Selection
currentSelection = randi(numel(fitness)); % In this demo, select one random shape
d.selection.models{hypID} = results{hypID}.models; % Save user model to use as constraint model

showPhenotype(genes(currentSelection,:),d,1.1,[]); title('Injected Perturbations of Selection');
fig(4) = gcf;

%%
phenotypes = d.getPhenotype(genes);
d.selection.selected{hypID} = features(currentSelection,:); 
d.selection.deselected{hypID} = features; d.selection.deselected{hypID}(currentSelection,:) = [];

hypID = hypID + 1;
newSamples = genes(currentSelection,:);
nNewPerSelected = ceil(pm.map.numInitSamples./length(currentSelection));
% Perturb selected shapes
for i=1:length(currentSelection)
    %newSampleMutations = pm.mutSelection * randn(nNewPerSelected,d.dof);
    newSampleMutations = 0.1 * randn(nNewPerSelected,d.dof);
    newSamples = [newSamples; genes(currentSelection(i),:) + newSampleMutations];
end
[newSamplesfitness,newSamplespolygons] = fitfun(newSamples,d); % Recalculate fitness! (User selection influences fitness values)
            
figure(99);plot(sort(newSamplesfitness));hold on;
showPhenotype(newSamples,d,1.1,[]); title('Injected Perturbations of Selection');

%% Run POEM's second iteration based on the user selection
[map{hypID}, config{hypID}, results{hypID}] = poem(newSamples,newSamplespolygons,newSamplesfitness,pm,d,2);
save([DOMAIN '_step2.mat']);
disp('HyperPref Step 2 Done');

%% Reload and extract results of third iteration and visualize
load([DOMAIN '_step2.mat']);
[genes,fitness,features,bins] = extractMap(results{2}.maps{1});
% Visualization
colors = [zeros(size(genes,1),1) fitness zeros(size(genes,1),1)];            
showPhenotype(genes, d, p.featureResolution(1), [], bins, colors); title('1st Iteration Result');

