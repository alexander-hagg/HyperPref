function vae = convDefault3(latentDim,resolution,numFilters,filterSize,stride)
%CONFIGUREVAE VAE is configured as in 
% Burgess, C. P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G., & Lerchner, A. (2017). 
% Understanding disentangling in ? -VAE
% 10 Apr 2018, (Nips).
%
imageSize = [resolution resolution 1];
weightsInitializer = 'glorot'; %he narrow-normal glorot

vae.encoderLG = layerGraph([
    imageInputLayer(imageSize,'Name','input_encoder','Normalization','none')
    convolution2dLayer(filterSize, numFilters, 'Padding','same', 'Stride', stride, 'Name', 'conv1', 'WeightsInitializer', weightsInitializer)
    reluLayer('Name','relu01')
    fullyConnectedLayer(2 * latentDim, 'Name', 'fc_encoder', 'WeightsInitializer', weightsInitializer)
    ]);

vae.decoderLG = layerGraph([
    imageInputLayer([1 1 latentDim],'Name','i','Normalization','none')
    transposedConv2dLayer(filterSize, numFilters, 'Cropping', 'same', 'Stride', stride, 'Name', 'transpose1', 'WeightsInitializer', weightsInitializer)
    reluLayer('Name','relu11')
    ]);

end

