addpath(genpath(pwd))                           % Set path to all modules
DOF = 16;                                       % Degrees of freedom, Catmull-Rom spline domain
d = domain(DOF);                                % Domain configuration
FITNESSFUNCTION = 'bmpSymmetry'; rmpath(genpath('domain/catmullRom/fitnessFunctions')); addpath(genpath(['domain/catmullRom/fitnessFunctions/' FITNESSFUNCTION]));
numInitSamples = 512;
sobSequence = scramble(sobolset(d.dof,'Skip',1e3),'MatousekAffineOwen');  sobPoint = (1-1)*numInitSamples+1;
observations = range(d.ranges').*sobSequence(sobPoint:(sobPoint+numInitSamples)-1,:)+d.ranges(:,1)';
[~,phenotypes] = fitfun(observations,d);

system(['rm -rf ' 'data/workdir' '/*']);
writePhenoToDisk(phenotypes,'data/workdir');

imds = imageDatastore('data/workdir/', ...
    'IncludeSubfolders',false, ...
    'LabelSource','foldernames');

%[data,imds] = getDataPoly(phenotypes,'data/workdir',32,0.9);
imds = shuffle(imds);
[imdsTrain,imdsVal,imdsTest] = splitEachLabel(imds,0.95,0.025);

dsTrainNoisy = transform(imdsTrain,@addNoise);
dsValNoisy = transform(imdsVal,@addNoise);
dsTestNoisy = transform(imdsTest,@addNoise);

dsTrain = combine(dsTrainNoisy,imdsTrain);
dsVal = combine(dsValNoisy,imdsVal);
dsTest = combine(dsTestNoisy,imdsTest);

dsTrain = transform(dsTrain,@commonPreprocessing);
dsVal = transform(dsVal,@commonPreprocessing);
dsTest = transform(dsTest,@commonPreprocessing);

dsTrain = transform(dsTrain,@augmentImages);



imageLayer = imageInputLayer([64,64,1]);

encodingLayers = [ ...
    convolution2dLayer(3,16,'Padding','same'), ...
    reluLayer, ...
    maxPooling2dLayer(2,'Padding','same','Stride',2), ...
    convolution2dLayer(3,8,'Padding','same'), ...
    reluLayer, ...
    maxPooling2dLayer(2,'Padding','same','Stride',2), ...
    convolution2dLayer(3,8,'Padding','same'), ...
    reluLayer, ...
    maxPooling2dLayer(2,'Padding','same','Stride',2)];

decodingLayers = [ ...
    createUpsampleTransponseConvLayer(2,8), ...
    reluLayer, ...
    createUpsampleTransponseConvLayer(2,8), ...
    reluLayer, ...
    createUpsampleTransponseConvLayer(2,16), ...
    reluLayer, ...
    convolution2dLayer(3,1,'Padding','same'), ...
    clippedReluLayer(1.0), ...
    regressionLayer];

layers = [imageLayer,encodingLayers,decodingLayers];

%%
options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize',imds.ReadSize, ...
    'Shuffle','never', ...
    'Plots','training-progress', ...
    'Verbose',true);
    'ValidationData',dsVal, ...

net = trainNetwork(dsTrain,layers,options);

%%
function dataOut = addNoise(data)
dataOut = data;
for idx = 1:size(data,1)
    dataOut{idx} = imnoise(data{idx},'salt & pepper');
end
end

function dataOut = commonPreprocessing(data)
dataOut = cell(size(data));
for col = 1:size(data,2)
    for idx = 1:size(data,1)
        temp = single(data{idx,col});
        temp = imresize(temp,[64,64]);
        temp = rescale(temp);
        dataOut{idx,col} = temp;
    end
end
end

function dataOut = augmentImages(data)
dataOut = cell(size(data));
for idx = 1:size(data,1)
    rot90Val = randi(4,1,1)-1;
    dataOut(idx,:) = {rot90(data{idx,1},rot90Val),rot90(data{idx,2},rot90Val)};
end
end

function out = createUpsampleTransponseConvLayer(factor,numFilters)

filterSize = 2*factor - mod(factor,2);
cropping = (factor-mod(factor,2))/2;
numChannels = 1;

out = transposedConv2dLayer(filterSize,numFilters, ...
    'NumChannels',numChannels,'Stride',factor,'Cropping',cropping);
end