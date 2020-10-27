clear;
%% Configuration
addpath(genpath(pwd))                           % Set path to all modules
DOF = 16;DOMAIN = 'catmullRom';                 % Degrees of freedom, Catmull-Rom spline domain
d = domain(DOF);                                % Domain configuration
numLatentDims = 2;
m = cfgLatentModel('data/workdir',d.resolution, numLatentDims);% VAE configuration

% Create shapes variations
shapeParams = [1 0.3 1 0.3 1 0.3 1 0.3 zeros(1,8)];
numShapes = 20; scaling = [0.2 1.0]; rotation = [0 0.45*pi];
genomes = createShapeVariations(shapeParams,numShapes,scaling,rotation);

%% Three datasets: last two miss part of the training data
deSelect{1} = []; deSelect{2} = [8:13]; deSelect{3} = [14:20];
allGenomes{1} = reshape(genomes,[],d.dof);
allGenomes{2} = genomes;allGenomes{2}(deSelect{2},deSelect{2},:) = nan;allGenomes{2} = reshape(allGenomes{2},[],d.dof);allGenomes{2}(all(isnan(allGenomes{2})'),:) = [];
allGenomes{3} = genomes;allGenomes{3}(deSelect{3},deSelect{3},:) = nan;allGenomes{3} = reshape(allGenomes{3},[],d.dof);allGenomes{3}(all(isnan(allGenomes{3})'),:) = [];

% Produce phenotypes
x = 1:numShapes; y = x; [X,Y] = ndgrid(x,y);
for i=1:3
    % Get phenotypes
    [fitness{i},phenotypes{i}] = fitfun(allGenomes{i},d);
    
    % Adjust placement to missing shapes
    tX = X; tY = Y; tX(deSelect{i},deSelect{i}) = nan; tY(deSelect{i},deSelect{i}) = nan;
    placement = [tX(~isnan(tX)'),tY(~isnan(tY)')];
    fig(i) = figure(i); hold off; ax = gca;
    showPhenotype(allGenomes{i},d,1.2,ax,placement); axis equal;
    axis([0 1.3*numShapes -1.3*numShapes 0]);
end

%% Train models
for i=1:3; allModels{i} = trainFeatures(phenotypes{i},m);end
save([DOMAIN '_bvae_1000_3.mat']);

%% Analysis: sample all models
for i=1:3
    % Get predicted features for *all* shapes, incl. the ones missing from
    % training data for particular model.
    [~,phen] = getPhenotypeBoolean(phenotypes{i}, allModels{i}.encoderLG.Layers(1).InputSize(1));
    features{i} = getPrediction(phen,allModels{i});
    
    % Create latent samples
    minFeatures = 1*min(features{i}(:)); maxFeatures = 1*max(features{i}(:));
    nSamples = 10;
    x = minFeatures:(maxFeatures-minFeatures)/nSamples:maxFeatures; y = x; [X,Y] = ndgrid(x,y);
    varyCoords = [X(:),Y(:)]';
    input = []; input(1,1,:,:) = varyCoords; input = dlarray(input,'SSCB');
    genImgSample{i} = sigmoid(predict(allModels{i}.decoderNet, input));
    genImgSample{i} = gather(extractdata(genImgSample{i}));
    % Place collected VAE outputs in latent space
    scale = d.resolution;
    [normVaryCoords{i},mapping{i}] = mapminmax(varyCoords,-nSamples,nSamples);
    bitmapCoords = 1 + (ceil(scale*normVaryCoords{i})-min(ceil(scale*normVaryCoords{i}(:))));
    
    % Get reproduced shapes from training data
    input = []; input(1,1,:,:) = features{i}'; input = dlarray(input,'SSCB');
    genImgTrain{i} = sigmoid(predict(allModels{i}.decoderNet, input));
    genImgTrain{i} = gather(extractdata(genImgTrain{i}));    %reproduced
    
    % Place missing training data, iff exists
    if ~isempty(deSelect{i})
        missingGenomes = genomes(deSelect{i},deSelect{i},:); missingGenomes = reshape(missingGenomes,[],d.dof);
        [~,phen] = fitfun(missingGenomes,d);
        [~,phen] = getPhenotypeBoolean(phen, allModels{i}.encoderLG.Layers(1).InputSize(1));
        missingFeatures{i} = getPrediction(phen,allModels{i});
        input = []; input(1,1,:,:) = missingFeatures{i}'; input = dlarray(input,'SSCB');
        genImgMissing{i} = sigmoid(predict(allModels{i}.decoderNet, input));
        genImgMissing{i} = gather(extractdata(genImgMissing{i}));
    end
    
    % Place *original* shapes into latent space
    [~,phen] = fitfun(allGenomes{1},d);
    [~,phen] = getPhenotypeBoolean(phen, allModels{i}.encoderLG.Layers(1).InputSize(1));
    allFeatures{i} = getPrediction(phen,allModels{i});
end

%% RQI Analysis: calculate reconstruction errors of missing shapes (relative hamming error)
for i=1:3
    [~,phen] = fitfun(reshape(allGenomes{i},[],d.dof),d);
    [~,phen] = getPhenotypeBoolean(phen, allModels{i}.encoderLG.Layers(1).InputSize(1));
    
    for jj=1:length(phen)
        originalPhen = phen{jj};
        reconstructedPhen = imbinarize(genImgTrain{i}(:,:,1,jj));
        hamming{i,1}(jj) = sum(xor(originalPhen(:),reconstructedPhen(:)));
    end
    relHammingError{i,1} = hamming{i,1}/(64*64);
    meanHammingError(i,1) = mean(relHammingError{i,1});
    
    if ~isempty(deSelect{i})
        missingGenomes = genomes(deSelect{i},deSelect{i},:); missingGenomes = reshape(missingGenomes,[],d.dof);
        [~,phen] = fitfun(missingGenomes,d);
        [~,phen] = getPhenotypeBoolean(phen, allModels{i}.encoderLG.Layers(1).InputSize(1));
        
        for jj=1:length(phen)
            originalPhen = phen{jj};
            reconstructedPhen = imbinarize(genImgMissing{i}(:,:,1,jj));
            hamming{i,2}(jj) = sum(xor(originalPhen(:),reconstructedPhen(:)));
        end
        relHammingError{i,2} = hamming{i,2}/(64*64);
        meanHammingError(i,2) = mean(relHammingError{i,2});
    end
    
end

disp(['Rel. Hamming errors: ']);
meanHammingError

%% RQII Analysis: calculate latent distance of missing shapes to all shapes

% get distances between shapes in model A
[~,phen] = fitfun(reshape(allGenomes{1},[],d.dof),d);
[~,phen] = getPhenotypeBoolean(phen, allModels{1}.encoderLG.Layers(1).InputSize(1));
allTrainingFeatures = getPrediction(phen,allModels{1});
distances = pdist2(allTrainingFeatures,allTrainingFeatures);
distances(1:size(distances,1)+1:end) = nan; % Leave out distances from elements to themselves
minDistances(1,1) = min(distances(:));
    
clear fig;
for i=2:3
    [~,phen] = fitfun(reshape(allGenomes{i},[],d.dof),d);
    [~,phen] = getPhenotypeBoolean(phen, allModels{i}.encoderLG.Layers(1).InputSize(1));
    allTrainingFeatures = getPrediction(phen,allModels{i});
    
    missingGenomes = genomes(deSelect{i},deSelect{i},:); missingGenomes = reshape(missingGenomes,[],d.dof);
    [~,phen] = fitfun(missingGenomes,d);
    [~,phen] = getPhenotypeBoolean(phen, allModels{i}.encoderLG.Layers(1).InputSize(1));
    allMissingFeatures = getPrediction(phen,allModels{i});
    
    
    BMIN = 0; BMAX = 5; YMAX = 0.3;
    fig((i-2)*2+1) = figure;
    subplot(1,3,1);
    distances = pdist2(allTrainingFeatures,allTrainingFeatures);
    distances(1:size(distances,1)+1:end) = nan; % Leave out distances from elements to themselves
    minDistances(i,1) = min(distances(:));
    histogram(distances(:),20,'BinLimits',[BMIN,BMAX],'Normalization','probability')
    ax = gca;ax.YLim = [0 YMAX];grid on;
    title('train-train');
    
    subplot(1,3,2);
    distances = pdist2(allMissingFeatures,allMissingFeatures);
    distances(1:size(distances,1)+1:end) = nan; % Leave out distances from elements to themselves
    minDistances(i,2) = min(distances(:));
    histogram(distances,20,'BinLimits',[BMIN,BMAX],'Normalization','probability')
    ax = gca;ax.YLim = [0 YMAX];grid on;
    title('missing-missing');
    
    subplot(1,3,3);
    distances = pdist2(allMissingFeatures,allTrainingFeatures);
    minDistances(i,3) = min(distances(:));
    histogram(distances,20,'BinLimits',[BMIN,BMAX],'Normalization','probability')
    ax = gca;ax.YLim = [0 YMAX];grid on;
    title('missing-train');
    
    fig((i-2)*2+2) = figure;
    hold off;
    scatter(allTrainingFeatures(:,1),allTrainingFeatures(:,2),32,'b','filled');
    hold on;
    scatter(allMissingFeatures(:,1),allMissingFeatures(:,2),32,'r','filled');
    axis equal;
    ax = gca;
    ax.XTickLabel = [];
    ax.YTickLabel = [];
end

disp(['Min. latent distances: ']);
minDistances
%save_figures(fig, '.', 'IDNODN_Analysis', 12, [7 5])


%% Visualization
for i=1:3
    
    %% Turn VAE outputs to viewable images
    % Create image with samples in latent coordinates
    imgSize = [0 -min(bitmapCoords(:)) + max(bitmapCoords(:))+scale];
    clear img; img{1} = (zeros(range(imgSize),range(imgSize)));img{2} = img{1}; img{3} = img{1}; img{4} = img{1};
    for jj=1:size(genImgSample{i},4)
        coords = bitmapCoords(:,jj)';
        img{1}([coords(1):(coords(1)+scale-1)],[coords(2):(coords(2)+scale-1)]) = img{1}([coords(1):(coords(1)+scale-1)],[coords(2):(coords(2)+scale-1)]) + squeeze(genImgSample{i}(:,:,:,jj));
    end
    img{1} = img{1} > 0.9;
    
    % Create image with training examples in latent coordinates
    normTrainCoords = mapminmax.apply(features{i}',mapping{i})';
    trainCoords = 1 + (ceil(scale*normTrainCoords)-min(ceil(scale*normVaryCoords{i}(:))));
    for jj=1:size(genImgTrain{i},4)
        coords = trainCoords(jj,:);
        img{2}([coords(1):(coords(1)+scale-1)],[coords(2):(coords(2)+scale-1)]) = img{2}([coords(1):(coords(1)+scale-1)],[coords(2):(coords(2)+scale-1)]) + squeeze(genImgTrain{i}(:,:,:,jj));
    end
    img{2} = img{2} > 0.9;
    
    % Create image with missing training examples in latent coordinates
    if ~isempty(deSelect{i})
        normTrainCoords = mapminmax.apply(missingFeatures{i}',mapping{i})';
        trainCoords = 1 + (ceil(scale*normTrainCoords)-min(ceil(scale*normVaryCoords{i}(:))));
        for jj=1:size(genImgMissing{i},4)
            coords = trainCoords(jj,:);
            img{3}([coords(1):(coords(1)+scale-1)],[coords(2):(coords(2)+scale-1)]) = img{3}([coords(1):(coords(1)+scale-1)],[coords(2):(coords(2)+scale-1)]) + squeeze(genImgMissing{i}(:,:,:,jj));
        end
    end
    img{3} = img{3} > 0.9;
    
    %% Create image with ground truth training examples in latent coordinates
    normTrainCoords = mapminmax.apply(allFeatures{i}',mapping{i})';
    trainCoords = 1 + (ceil(scale*normTrainCoords)-min(ceil(scale*normVaryCoords{i}(:))));
    tImg = img{4};
    [~,boolPhenotypes] = getPhenotypeBoolean(phenotypes{1}, allModels{i}.encoderLG.Layers(1).InputSize(1));
    for jj=1:size(allFeatures{i},1)
        I = double(boolPhenotypes{jj}); BW = imbinarize(I); [B,L] = bwboundaries(BW,'noholes');
        shape = zeros(size(boolPhenotypes{jj}));
        for pix=1:length(B{1}(:,1))
            shape(B{1}(pix,1),B{1}(pix,2)) = 1;
        end
        shape = imdilate(shape, strel('disk', 1));
        coords = trainCoords(jj,:);
        tImg([coords(1):(coords(1)+scale-1)],[coords(2):(coords(2)+scale-1)]) = tImg([coords(1):(coords(1)+scale-1)],[coords(2):(coords(2)+scale-1)]) + shape;
    end
    img{4} = tImg > 0.9;
    
    % Add images together
    imgComplete = zeros(size(img{1}));
    imgComplete = 0.25*img{1};                             % Samples
    imgComplete(img{2}(:)>0) = 0.5*img{2}(img{2}(:)>0);    % Selected training examples
    imgComplete(img{3}(:)>0) = 0.75*img{3}(img{3}(:)>0);   % Missing training examples
    imgComplete(img{4}(:)>0) = 1*img{4}(img{4}(:)>0);      % Ground truth of all training examples
    
    % Show image
    fig(99+i) = figure(99+i);
    hold off; ax=gca;
    imshow(imgComplete);
    hold on;
    colormap([1 1 1; 0 0 0; 0 1 0; 1 0 0; 0 0 0]);
    drawnow
end

%%
save_figures(fig, '.', 'IDNODN2', 12, [5 5])





