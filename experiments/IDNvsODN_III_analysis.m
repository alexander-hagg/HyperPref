clear;
%% Configuration
addpath(genpath(pwd))                           % Set path to all modules
DOF = 16;                                       % Degrees of freedom, Catmull-Rom spline domain
d = domain(DOF);                                % Domain configuration
FITNESSFUNCTION = 'bmpSymmetry'; rmpath(genpath('domain/catmullRom/fitnessFunctions')); addpath(genpath(['domain/catmullRom/fitnessFunctions/' FITNESSFUNCTION]));

%% Visualize
BMIN = 0.5; BMAX = 1; clear counts fitnesses score

for replicate=1%:5
    disp(['Loading replicate: ' int2str(replicate)]);
    fileName = ['catmullRom_III'];
    load(['' fileName '_replicate_' int2str(replicate) '.mat']);
    
    for rep=1:length(latentDOFs)
        disp(['Latent DOF: ' int2str(latentDOFs(rep))]);
        
        % For fitness histograms
        counts(replicate,1,rep,:) = histcounts(map{rep,1}.fitness(:),20,'BinLimits',[BMIN,BMAX],'Normalization','count');
        counts(replicate,2,rep,:) = histcounts(map{rep,2}.fitness(:),20,'BinLimits',[BMIN,BMAX],'Normalization','count');
        counts(replicate,3,rep,:) = histcounts(map{rep,3}.fitness(:),20,'BinLimits',[BMIN,BMAX],'Normalization','count');
        counts(replicate,4,rep,:) = histcounts(map{rep,4}.fitness(:),20,'BinLimits',[BMIN,BMAX],'Normalization','count');
        
        % Get total fitness ("QD-score")
        fitnesses(replicate,rep,1) = sum(map{rep,1}.fitness);
        fitnesses(replicate,rep,2) = sum(map{rep,2}.fitness);
        fitnesses(replicate,rep,3) = sum(map{rep,3}.fitness);
        fitnesses(replicate,rep,4) = sum(map{rep,4}.fitness);
        
        % Get PD metrics
        genes = reshape(map{rep,1}.genes,[],d.dof); genes = genes(all(~isnan(genes)'),:);
        [~,flatbitmaps] = d.getPhenotype(genes);
        [score(replicate,rep,1)] = metricPD(flatbitmaps, 'hamming');
        
        genes = reshape(map{rep,2}.genes,[],latentDOFs(rep)); genes = genes(all(~isnan(genes)'),:);
        bitmaps = sampleVAE(genes,results{rep,1}.models(1).decoderNet);
        flatbitmaps = [];
        for i=1:length(bitmaps)
            bitmap = repairVAEoutput(bitmaps{i});
            flatbitmaps(i,:) = bitmap(:);
        end
        [score(replicate,rep,2)] = metricPD(flatbitmaps, 'hamming');
        
        genes = reshape(map{rep,3}.genes,[],d.dof); genes = genes(all(~isnan(genes)'),:);
        [~,flatbitmaps] = d.getPhenotype(genes);
        [score(replicate,rep,3)] = metricPD(flatbitmaps, 'hamming');
        
        genes = reshape(map{rep,4}.genes,[],latentDOFs(rep)); genes = genes(all(~isnan(genes)'),:);
        bitmaps = sampleVAE(genes,results{rep,3}.models(1).decoderNet);
        flatbitmaps = [];
        for i=1:length(bitmaps)
            bitmap = repairVAEoutput(bitmaps{i});
            flatbitmaps(i,:) = bitmap(:);
        end
        [score(replicate,rep,4)] = metricPD(flatbitmaps, 'hamming');
        
        losses(replicate,rep,1,:) = [results{rep,1}.models(1).statistics.loss(1) results{rep,1}.models(1).statistics.loss(50:50:end)];
        reconstructionLosses(replicate,rep,1,:) = [results{rep,1}.models(1).statistics.reconstructionLoss(1) results{rep,1}.models(1).statistics.reconstructionLoss(50:50:end)];
        regTerms(replicate,rep,1,:) = [results{rep,1}.models(1).statistics.regTerm(1) results{rep,1}.models(1).statistics.regTerm(50:50:end)];
        
        losses(replicate,rep,2,:) = [results{rep,3}.models(1).statistics.loss(1) results{rep,3}.models(1).statistics.loss(50:50:end)];
        reconstructionLosses(replicate,rep,2,:) = [results{rep,3}.models(1).statistics.reconstructionLoss(1) results{rep,3}.models(1).statistics.reconstructionLoss(50:50:end)];
        regTerms(replicate,rep,2,:) = [results{rep,3}.models(1).statistics.regTerm(1) results{rep,3}.models(1).statistics.regTerm(50:50:end)];
        
    end
end


%% Losses
fig(199) = figure(199)
subplot(3,2,1);hold off;
semilogy(prctile(reshape(reconstructionLosses(:,:,1,:),20,201),90));
hold on;
semilogy(squeeze(median(reconstructionLosses(:,:,1,:),[1 2 3])));
semilogy(prctile(reshape(reconstructionLosses(:,:,1,:),20,201),10));
ax = gca;ax.XTick = 0:50:200; ax.XTickLabel = ax.XTick*50;ylabel('Reconstruction Loss Term');xlabel('Epochs');grid on;
ax.YLim = [10^0 10^4];

subplot(3,2,2);hold off;
semilogy(prctile(reshape(reconstructionLosses(:,:,2,:),20,201),90));
hold on;
semilogy(squeeze(median(reconstructionLosses(:,:,2,:),[1 2 3])));
semilogy(prctile(reshape(reconstructionLosses(:,:,2,:),20,201),10));
ax = gca;ax.XTick = 0:50:200; ax.XTickLabel = ax.XTick*50;ylabel('Reconstruction Loss Term');xlabel('Epochs');grid on;
ax.YLim = [10^0 10^4];
%

subplot(3,2,3);hold off;
semilogy(prctile(reshape(regTerms(:,:,1,:),20,201),90));
hold on;
semilogy(squeeze(median(regTerms(:,:,1,:),[1 2 3])));
semilogy(prctile(reshape(regTerms(:,:,1,:),20,201),10));
ax = gca;ax.XTick = 0:50:200; ax.XTickLabel = ax.XTick*50;ylabel('KL Loss Term');xlabel('Epochs');grid on;
ax.YLim = [10^0 10^2];

subplot(3,2,4);hold off;
semilogy(prctile(reshape(regTerms(:,:,2,:),20,201),90));
hold on;
semilogy(squeeze(median(regTerms(:,:,2,:),[1 2 3])));
semilogy(prctile(reshape(regTerms(:,:,2,:),20,201),10));
ax = gca;ax.XTick = 0:50:200; ax.XTickLabel = ax.XTick*50;ylabel('KL Loss Term');xlabel('Epochs');grid on;
ax.YLim = [10^0 10^2];
%

subplot(3,2,5);hold off;
semilogy(prctile(reshape(losses(:,:,1,:),20,201),90));
hold on;
semilogy(squeeze(median(losses(:,:,1,:),[1 2 3])));
semilogy(prctile(reshape(losses(:,:,1,:),20,201),10));
ax = gca;ax.XTick = 0:50:200; ax.XTickLabel = ax.XTick*50;ylabel('Total \beta-Loss Term');xlabel('Epochs');grid on;
ax.YLim = [10^0 10^4];

subplot(3,2,6);hold off;
semilogy(prctile(reshape(losses(:,:,2,:),20,201),90));
hold on;
semilogy(squeeze(median(losses(:,:,2,:),[1 2 3])));
semilogy(prctile(reshape(losses(:,:,2,:),20,201),10));
ax = gca;ax.XTick = 0:50:200; ax.XTickLabel = ax.XTick*50;ylabel('Total \beta-Loss Term');xlabel('Epochs');grid on;
ax.YLim = [10^0 10^4];

save_figures(fig, '.', 'IDNODNIII-losses', 12, [5 5])

%% Histograms
YMAX = 70;
fig(1) = figure(1);
edges = BMIN:(BMAX-BMIN)./(20):BMAX;
subplot(4,1,1);
histogram('BinEdges',edges,'BinCounts',mean(squeeze(mean(squeeze(counts(:,1,:,:)),1)),1))

title('Parameter Search, Random')
ax = gca;ax.YLim = [0 YMAX];grid on;xlabel('fitness');ylabel('count');

subplot(4,1,2);
histogram('BinEdges',edges,'BinCounts',mean(squeeze(mean(squeeze(counts(:,2,:,:)),1)),1))
title('Latent Search, Random')
ax = gca;ax.YLim = [0 YMAX];grid on;xlabel('fitness');ylabel('count');

subplot(4,1,3);
histogram('BinEdges',edges,'BinCounts',mean(squeeze(mean(squeeze(counts(:,3,:,:)),1)),1))
title('Parameter Search, Continue')
ax = gca;ax.YLim = [0 YMAX];grid on;xlabel('fitness');ylabel('count');

subplot(4,1,4);
histogram('BinEdges',edges,'BinCounts',mean(squeeze(mean(squeeze(counts(:,4,:,:)),1)),1))
title('Latent Search, Continue')
ax = gca;ax.YLim = [0 YMAX];grid on;xlabel('fitness');ylabel('count');

%% Visualization
fig(2) = figure(2);
plot(repmat(latentDOFs',1,4), squeeze(mean(score,1)),'o-')
legend('Parameter, from random', 'Latent, from random', 'Parameter, continuation','Latent, continuation','Location','SouthEast')
xlabel('Latent DOF');
ylabel('Pure Diversity');
ax=gca;ax.XTick = latentDOFs;
ax.YAxis.Limits = [0 20];
grid on
title('Diversity of solution sets');

fig(3) = figure(3);
plot(repmat(latentDOFs',1,4), squeeze(mean(fitnesses,1))./pm.map.numInitSamples,'o-')
legend('Parameter, from random', 'Latent, from random', 'Parameter, continuation','Latent, continuation','Location','NorthEast')
xlabel(' Latent DOF');
ylabel('Avg. Fitness');
ax=gca;ax.XTick = latentDOFs;
ax.YAxis.Limits = [0.75 1];
grid on
title('Avg. Fitness');

save_figures(fig, '.', 'IDNODNIII', 12, [5 5])

%% Show some shapes
rep = 1;

genes = reshape(map{rep,1}.genes,[],d.dof); genes = genes(all(~isnan(genes)'),:);
fig(5) = figure(5);
bitmaps{1} = showPhenotypeBMP(genes,d,fig(5));

genes = reshape(map{rep,2}.genes,[],latentDOFs(rep)); genes = genes(all(~isnan(genes)'),:);
bitmapsVAE = sampleVAE(genes,results{rep,2}.models(1).decoderNet);
fig(6) = figure(6);
bitmaps{2} = showPhenotypeBMP(bitmapsVAE,latentDomain{replicate,rep,2},fig(6));

genes = reshape(map{rep,3}.genes,[],d.dof); genes = genes(all(~isnan(genes)'),:);
bitmapsPAR = d.getPhenotype(genes);
fig(7) = figure(7);
bitmaps{3} = showPhenotypeBMP(genes,d,fig(7));

genes = reshape(map{rep,4}.genes,[],latentDOFs(rep)); genes = genes(all(~isnan(genes)'),:);
bitmapsVAE = sampleVAE(genes,results{rep,4}.models(1).decoderNet);
fig(8) = figure(8);
bitmaps{4} = showPhenotypeBMP(bitmapsVAE,latentDomain{replicate,rep,4},fig(8));

save_figures(fig, '.', ['QDresults_latDim' int2str(latentDOFs(rep)) '_'], 12, [5 5])

%%
flatbmps = [];
for i=1:4
    for j=1:length(bitmaps{i})
        flatbmps(end+1,:) = bitmaps{i}{j}(:);
    end
end

cmap = parula(4);
tsneCoords = tsne(flatbmps,'Standardize',true,'Perplexity',30,'NumDimensions',3);
%%
figure(1);hold off;
scatter3(tsneCoords(1:256,1),tsneCoords(1:256,2),tsneCoords(1:256,3),32,cmap(1,:),'filled');hold on;
scatter3(tsneCoords(257:512,1),tsneCoords(257:512,2),tsneCoords(257:512,3),32,cmap(2,:),'filled');hold on;
legend('rng par','rng lat');
figure(2);hold off;
scatter3(tsneCoords(513:768,1),tsneCoords(513:768,2),tsneCoords(513:768,3),32,cmap(3,:),'filled');hold on;
scatter3(tsneCoords(769:1024,1),tsneCoords(769:1024,2),tsneCoords(769:1024,3),32,cmap(4,:),'filled');hold on;
legend('2nd par','2nd lat');





