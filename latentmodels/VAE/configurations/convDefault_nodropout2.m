function vae = convDefault_nodropout2(latentDim,resolution,numFilters,filterSize,stride)
%CONFIGUREVAE VAE is configured as in
% Burgess, C. P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G., & Lerchner, A. (2017).
% Understanding disentangling in ? -VAE
% 10 Apr 2018, (Nips).
%
imageSize = [resolution resolution 1];
weightsInitializer = 'glorot'; %he narrow-normal glorot

stride = 1;

vae.encoderLG = layerGraph([
    imageInputLayer(imageSize,'Name','input_encoder','Normalization','none')
    convolution2dLayer(filterSize, numFilters/8, 'Padding','same', 'Stride', stride, 'Name', 'conv1', 'WeightsInitializer', weightsInitializer)
    reluLayer('Name','relu01')
    maxPooling2dLayer(2,'Padding','same','Stride',2,'Name','maxpool01')
    convolution2dLayer(filterSize, numFilters/4, 'Padding','same', 'Stride', stride, 'Name', 'conv2', 'WeightsInitializer', weightsInitializer)
    reluLayer('Name','relu02')
    maxPooling2dLayer(2,'Padding','same','Stride',2,'Name','maxpool02')
    convolution2dLayer(filterSize, numFilters, 'Padding','same', 'Stride', stride, 'Name', 'conv3', 'WeightsInitializer', weightsInitializer)
    reluLayer('Name','relu03')
    maxPooling2dLayer(2,'Padding','same','Stride',2,'Name','maxpool03')
    fullyConnectedLayer(2 * latentDim, 'Name', 'fc_encoder', 'WeightsInitializer', weightsInitializer)
    ]);

vae.decoderLG = layerGraph([
    imageInputLayer([1 1 latentDim],'Name','i','Normalization','none')
    createUpsampleTransponseConvLayer(4,numFilters,'conv11')
    %transposedConv2dLayer(filterSize, numFilters, 'Cropping', 'same', 'Stride', stride, 'Name', 'transpose1', 'WeightsInitializer', weightsInitializer)
    reluLayer('Name','relu11')
    createUpsampleTransponseConvLayer(4,numFilters/4,'conv12')
    %transposedConv2dLayer(filterSize, numFilters/4, 'Cropping', 'same', 'Stride', stride, 'Name', 'transpose2', 'WeightsInitializer', weightsInitializer)
    reluLayer('Name','relu12')
    createUpsampleTransponseConvLayer(4,numFilters/8,'conv13')
    %transposedConv2dLayer(filterSize, numFilters/8, 'Cropping', 'same', 'Stride', stride, 'Name', 'transpose3', 'WeightsInitializer', weightsInitializer)
    reluLayer('Name','relu13')
    %createUpsampleTransponseConvLayer(4,numFilters,'conv14')
    transposedConv2dLayer(filterSize, 1, 'Cropping', 'same', 'Name', 'transpose5', 'WeightsInitializer', weightsInitializer)
    reluLayer('Name','relu14')
    %clippedReluLayer(1.0,'Name','relu14')
    ]);
end

function out = createUpsampleTransponseConvLayer(factor,numFilters,name)
weightsInitializer = 'glorot'; %he narrow-normal glorot

filterSize = 2*factor - mod(factor,2);
cropping = (factor-mod(factor,2))/2;
numChannels = 1;
out = transposedConv2dLayer(filterSize,numFilters, ...
    'NumChannels',numChannels,'Stride',factor,'Cropping',cropping,'Name',name,'WeightsInitializer',weightsInitializer);
end

