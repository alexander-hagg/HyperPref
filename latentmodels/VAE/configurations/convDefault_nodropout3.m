function vae = convDefault2(latentDim,resolution,numFilters,filterSize,stride)
%CONFIGUREVAE VAE is configured as in 
% Burgess, C. P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G., & Lerchner, A. (2017). 
% Understanding disentangling in ? -VAE
% 10 Apr 2018, (Nips).
%
imageSize = [resolution resolution 1];
weightsInitializer = 'glorot'; %he narrow-normal glorot

vae.encoderLG = layerGraph([
    imageInputLayer(imageSize,'Name','input_encoder','Normalization','none')
    convolution2dLayer(filterSize, numFilters/8, 'Padding','same', 'Stride', stride, 'Name', 'conv1', 'WeightsInitializer', weightsInitializer)
    reluLayer('Name','relu01')
    convolution2dLayer(filterSize, numFilters/4, 'Padding','same', 'Stride', stride, 'Name', 'conv2', 'WeightsInitializer', weightsInitializer)
    reluLayer('Name','relu02')
    convolution2dLayer(filterSize, numFilters, 'Padding','same', 'Stride', stride, 'Name', 'conv3', 'WeightsInitializer', weightsInitializer)
    reluLayer('Name','relu03')
    fullyConnectedLayer(2 * latentDim, 'Name', 'fc_encoder', 'WeightsInitializer', weightsInitializer)
    ]);

vae.decoderLG = layerGraph([
    imageInputLayer([1 1 latentDim],'Name','i','Normalization','none')
    %transposedConv2dLayer(filterSize, numFilters, 'Cropping', 'same', 'Stride', stride, 'Name', 'transpose1', 'WeightsInitializer', weightsInitializer)
    createUpsampleTransponseConvLayer(4,numFilters,'transpose1')
    reluLayer('Name','relu11')
    %transposedConv2dLayer(filterSize, numFilters/4, 'Cropping', 'same', 'Stride', stride, 'Name', 'transpose2', 'WeightsInitializer', weightsInitializer)
    createUpsampleTransponseConvLayer(4,numFilters/4,'transpose2')
    reluLayer('Name','relu12')
    %transposedConv2dLayer(filterSize, numFilters/8, 'Cropping', 'same', 'Stride', stride, 'Name', 'transpose3', 'WeightsInitializer', weightsInitializer)
    createUpsampleTransponseConvLayer(4,numFilters/8,'transpose3')
    reluLayer('Name','relu13')
    transposedConv2dLayer(filterSize, 1, 'Cropping', 'same', 'Name', 'transpose4', 'WeightsInitializer', weightsInitializer)
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
