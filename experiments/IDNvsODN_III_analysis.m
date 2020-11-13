clear;
%% Configuration
addpath(genpath(pwd))                           % Set path to all modules
DOF = 16;                                       % Degrees of freedom, Catmull-Rom spline domain
d = domain(DOF);                                % Domain configuration
FITNESSFUNCTION = 'bmpSymmetry'; rmpath(genpath('domain/catmullRom/fitnessFunctions')); addpath(genpath(['domain/catmullRom/fitnessFunctions/' FITNESSFUNCTION]));

%% Visualize
BMIN = 0.5; BMAX = 1; clear counts fitnesses score

for replicate=1:5
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
        
    end
end

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
title('Avg. Fitness - does not say much: local optima can have lower fitness');

save_figures(fig, '.', 'IDNODNIII', 12, [5 5])

%% Show some shapes
rep = 3;

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




