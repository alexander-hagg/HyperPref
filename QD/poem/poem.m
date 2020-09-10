function [map,configs,stats] = poem(observations,phenotypes,fitness,p,d,varargin)
%POEM - Phenotype Optimization through Evolution of a Manifold
% Main run script of POEM algorithm
%
% Author: Alexander Hagg
% Bonn-Rhein-Sieg University of Applied Sciences (HBRS)
% email: alexander.hagg@h-brs.de
% Nov 2019; Last revision: 12-Nov-2019

p.map.categorize = p.categorize;    
    
if p.display.illu
    figID = 2;
    if nargin > 5
        figID = varargin{1};
    end
    figPhenotypes = figure(figID);hold off;
end

for iter=1:p.numIterations
    
    % 1. Train VAE
    disp('Training latent model');
    p.map.model = trainFeatures(phenotypes,p.model);
    stats.models{iter} = p.map.model;
    
    features = p.categorize(observations,phenotypes,p.map,d);    
    if p.display.illu
        figure(figPhenotypes);
        subplot(2,p.numIterations+1,iter)
        fits = fitfun(observations,d);
        fitcolor = [0 1 0].*fits + [1 0 0].*(1-fits);
        fitcolor = discretize(fitcolor,0:0.25:1)./5;
        showPhenotype(observations,d,figPhenotypes,features,fitcolor);
        drawnow;
    end
    configs{iter} = p;
    
    % 2. Illuminate with QD
    disp('Illuminate latent space with QD');
    map = createMap(d, p.map);    
    [replaced, replacement] = nicheCompete(observations,fitness,map,d,p.map,features);
    map = updateMap(replaced,replacement,map,fitness,observations,features);
    map = illuminate(map,p.map,d,p.model);
    
    % 3. Select new model members
    disp('Select new members');
    candidates = reshape(map.genes,[],d.dof);
    candidates(all(isnan(candidates)'),:) = [];
    [fitness,phenotypes] = fitfun(candidates,d);
    candidates(all(isnan(fitness)'),:) = [];
    phenotypes(all(isnan(fitness)')) = [];
    fitness(all(isnan(fitness)')) = [];
    observations = candidates;
    
    % 4. Statistics
    disp('Get statistics');    
    stats.fitness.mean(iter) = nanmean(map.fitness(:)); stats.fitness.median(iter) = nanmedian(map.fitness(:)); stats.fitness.std(iter) = nanstd(map.fitness(:)); stats.fitness.total(iter) = nansum(map.fitness(:));
    stats.elites.number(iter) = sum(~isnan(map.fitness(:)));
    %stats.model.trainingLosses(iter,:) = p.map.model.losses;
    stats.maps{iter} = map;
    
    if p.display.illu
        features = p.categorize(observations,phenotypes,p.map,d); 
        figure(figPhenotypes);
        subplot(2,p.numIterations+1,p.numIterations+1 + iter)
        fits = fitfun(observations,d);
        fitcolor = [0 1 0].*fits + [1 0 0].*(1-fits);
        fitcolor = discretize(fitcolor,0:0.25:1)./5;
        showPhenotype(observations,d,figPhenotypes,features,fitcolor);
        drawnow;
    end
    
end


% 1. Train final VAE
disp('Train final VAE model');  
p.map.model = trainFeatures(phenotypes,p.model);
stats.models{iter+1} = p.map.model;
features = p.categorize(observations,phenotypes,p.map,d);

if p.display.illu
    figure(figPhenotypes);
    subplot(1,p.numIterations+1,iter+1)
    fits = fitfun(observations,d);
    fitcolor = [0 1 0].*fits + [1 0 0].*(1-fits);
    fitcolor = discretize(fitcolor,0:0.25:1)./5;
    showPhenotype(observations,d,figPhenotypes,features,fitcolor);
    drawnow;
end

end
