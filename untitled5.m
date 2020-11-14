[phens1,flatbitmaps1] = d.getPhenotype(map{rep,1}.genes);
phens2 = latentDomain{replicate,rep,2}.getPhenotype(map{rep,2}.genes);
flatbitmaps2 = [];
for i=1:length(phens2)
    bitmap = repairVAEoutput(phens2{i});
    imagesc(bitmap);drawnow;pause(0.1);
    flatbitmaps2(i,:) = bitmap(:);
end
        
metricPD(flatbitmaps1, 'hamming')
metricPD(flatbitmaps2, 'hamming')

%%
for i=1:length(phens1)
    ph = phens1{i};
    %ph = imbinarize(ph,0.9*max(ph(:)));
    imagesc(ph);
    drawnow;
    pause(0.2);
end