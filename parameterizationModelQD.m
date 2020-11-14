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
[fitness,phenotypes] = fitfun(observations,d);
        
clear model losses
% latentDim  numFilters filterSize stride
%numFilterss = [32 64 128]
numFilterss = [16 32 64]
filterSizes = [2 3 4]
for i = 1:length(numFilterss)
    for j = 1:length(filterSizes)
        numFilters = numFilterss(i)
        filterSize = filterSizes(j)
        % Run POEM on parameter space with latent space niching
        m = cfgLatentModel('data/workdir',d.resolution,latentDOFs(1),numFilters,filterSize);                    % VAE configuration
        model{i,j} = trainFeatures(phenotypes,m);
        losses(i,j,:) = model{i,j}.statistics.reconstructionLoss(model{i,j}.statistics.reconstructionLoss > 0);
    end
end
%%
clear minValLosses vallosses trainlosses
for i = 1:length(numFilterss)
    for j = 1:length(filterSizes)
        minValLosses(i,j) = model{i,j}.statistics.minValLoss;
        vallosses(i,j,:) = model{i,j}.statistics.loss;
        trainlosses(i,j,:) = [model{i,j}.statistics.training.loss(1) model{i,j}.statistics.training.loss(50:50:2000)];
        minValEpoch(i,j) = model{i,j}.statistics.minValLossEpoch;
    end
end

figure(1);
%imagesc(minValLosses)
hm = heatmap(minValLosses)
xlabel('filterSizes');
ylabel('numFilters');
hm.XDisplayLabels{1} = int2str(filterSizes(1));
hm.XDisplayLabels{2} = int2str(filterSizes(2));
hm.XDisplayLabels{3} = int2str(filterSizes(3));
hm.YDisplayLabels{1} = int2str(numFilterss(1));
hm.YDisplayLabels{2} = int2str(numFilterss(2));
hm.YDisplayLabels{3} = int2str(numFilterss(3));
%ax = gca;ax.XTick = 1:3;ax.XTickLabel = {filterSizes(:)}ax.YTick = 1:3;ax.YTickLabel = {numFilterss(:)}

%%
numX = ceil(sqrt(numel(phenotypes)));
numY = ceil(sqrt(numel(phenotypes)));
%latentVectors = randn(numX*numY,16); bitmaps = sampleVAE(latentVectors,model{3,3}.decoderNet);

[latent,xPred,xTrue] = getPrediction(phenotypes,model{2,3});

figure(2);
for i=1:numel(xTrue)
    subplot(1,2,1);
    imagesc(xTrue{i});
    title('true');
    subplot(1,2,2);
    imagesc(squeeze(xPred(:,:,:,i)));
    title('predicted');
    drawnow;
    pause(0.1);
end

%% show validation losses over the epochs
figure(3);hold off;
legendEntrees = {};
cmap = parula(3);
linStyles = {'-','--',':'};
pl = [];
for i = 1:length(numFilterss)
    for j = 1:length(filterSizes)
        k = 10;
        vals = movmean(squeeze(vallosses(i,j,:)),k);
        pl(end+1) = plot(vals,'Color',cmap(j,:),'LineWidth',2,'LineStyle',linStyles{i});
        hold on;
        plot([minValEpoch(i,j) minValEpoch(i,j)],[500 minValLosses(i,j)],'k');%'Color',cmap(j,:));
        legendEntrees{end+1} = ['#Filters: ' int2str(numFilterss(i)) ' - fSize: ' int2str(filterSizes(j))];
    end
end
legend(pl,legendEntrees);
title('Validation loss');


figure(4);hold off;
legendEntrees = {};
cmap = parula(3);
linStyles = {'-','--',':'};
pl = [];
for i = 1:length(numFilterss)
    for j = 1:length(filterSizes)
        k = 10;
        vals = movmean(squeeze(trainlosses(i,j,:)),k);
        pl(end+1) = plot([1 50:50:(size(vals,1)-1)*50],vals,'Color',cmap(j,:),'LineWidth',2,'LineStyle',linStyles{i});
        hold on;
        plot([minValEpoch(i,j) minValEpoch(i,j)],[0 minValLosses(i,j)],'k');%'Color',cmap(j,:));
        legendEntrees{end+1} = ['#Filters: ' int2str(numFilterss(i)) ' - fSize: ' int2str(filterSizes(j))];
    end
end
legend(pl,legendEntrees);
title('Training loss');

