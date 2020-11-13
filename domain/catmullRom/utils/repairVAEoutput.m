function filteredIMG = repairVAEoutput(bitmap)
%REPAIRVAEOUTPUT Summary of this function goes here
%   Detailed explanation goes here
if ~islogical(bitmap)
    bitmap = imbinarize(bitmap,0.9);
end
[B,L] = bwboundaries(bitmap,'noholes');
maxB = size(B{1},1); maxBID = 1;
for bb=1:length(B)
    if size(B{bb},1) > maxB;
        maxB = size(B{bb},1);
        maxBID = bb;
    end
end
filteredIMG = (L==maxBID);
filteredIMG = imfill(filteredIMG,'holes');
end

