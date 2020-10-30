function bitmaps = sampleVAE(latentVector,decoderNet)
input = reshape(latentVector,1,1,size(latentVector,2),[]);
input = dlarray(input,'SSCB');
generatedImage = sigmoid(predict(decoderNet, input));
returnImage = gather(extractdata(generatedImage));
for i=1:size(returnImage,4)
    bitmaps{i} = squeeze(returnImage(:,:,1,i));
end
end
