function vae = convDefault_matlabbook(latentDim,resolution,numFilters,filterSize,stride)
%CONFIGUREVAE VAE is configured as in 
% Burgess, C. P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G., & Lerchner, A. (2017). 
% Understanding disentangling in ? -VAE
% 10 Apr 2018, (Nips).
%
imageSize = [resolution resolution 1];
weightsInitializer = 'glorot'; %he narrow-normal glorot

vae.encoderLG = layerGraph([
    imageInputLayer(imageSize,'Name','input_encoder','Normalization','none')
    convolution2dLayer(3,16,'Padding','same','Name','conv01')
    reluLayer('Name','relu01')
    maxPooling2dLayer(2,'Padding','same','Stride',2,'Name','maxpool01')
    convolution2dLayer(3,8,'Padding','same','Name','conv02')
    reluLayer('Name','relu02')
    maxPooling2dLayer(2,'Padding','same','Stride',2,'Name','maxpool02')
    convolution2dLayer(3,8,'Padding','same','Name','conv03')
    reluLayer('Name','relu03')
    maxPooling2dLayer(2,'Padding','same','Stride',2,'Name','maxpool03')
    ]);

vae.decoderLG = layerGraph([
    imageInputLayer([1 1 latentDim],'Name','i','Normalization','none')
    createUpsampleTransponseConvLayer(4,8)
    reluLayer('Name','relu11')
    createUpsampleTransponseConvLayer(4,8)
    reluLayer('Name','relu12')
    createUpsampleTransponseConvLayer(4,16)
    reluLayer('Name','relu13')
    convolution2dLayer(3,1,'Padding','same','Name','conv14')
    clippedReluLayer(1.0,'Name','relu14')
    ]);

end

function out = createUpsampleTransponseConvLayer(factor,numFilters)
filterSize = 2*factor - mod(factor,2);
cropping = (factor-mod(factor,2))/2;
numChannels = 1;
out = transposedConv2dLayer(filterSize,numFilters, ...
    'NumChannels',numChannels,'Stride',factor,'Cropping',cropping,'Name',int2str(randi(500)));
end