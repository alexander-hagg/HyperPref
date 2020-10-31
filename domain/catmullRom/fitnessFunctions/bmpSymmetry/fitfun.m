function [fitness,phenotypes,rawFitness] = fitfun(input,d)
%fitfun - "ui compare to user selected shapes" fitness function
% Fitness is normalized between 0 and 1
%
% Syntax:  [fitness,phenotypes] = npolyObjective(genomes,d)
%
% Inputs:
%    genomes        - [NxM] - N genomes with dof = M
%    d              - cell - Domain configuration.
%
% Outputs:
%    fitness        - [Nx1] - Validation flags
%    phenotypes     - cell[Nx1] - phenotypes (to prevent recalculating
%                                 of phenotypes, we offer them back here
%
%
% Author: Alexander Hagg
% Bonn-Rhein-Sieg University of Applied Sciences (HBRS)
% email: alexander.hagg@h-brs.de
% Jul 2019; Last revision: 15-Aug-2019
%
%------------- BEGIN CODE --------------
if isempty(input); fitness = []; polygons = []; rawFitness = []; return; end

% Create bitmaps if input is in parameter space
if ~iscell(input)
    phenotypes = d.getPhenotype(input);
else
    phenotypes = input;
end

logicalPhenotypes = phenotypes;

for i=1:length(phenotypes)
    if ~islogical(phenotypes{i})
        logicalPhenotypes{i} = imbinarize(phenotypes{i},0.9);
    end
    B = bwboundaries(logicalPhenotypes{i},'noholes');
    id = 1; maxVal = size(B{1},1);
    if length(B) > 1
        for boundaryID=2:length(B)
            if size(B{boundaryID},1) > maxVal
                id = boundaryID; maxVal = size(B{boundaryID},1);
            end
        end
    end
    
    boundary = B{id};
    %normalize boundary coordinates for fitness function
    normBoundary = mapminmax(boundary',-1,1)';
    if ~mod(size(normBoundary,1),2)==0; normBoundary(end,:) = []; end % Make even number # points
    a = normBoundary(1:ceil(end/2),:); % Take first half of points
    x = normBoundary(ceil(end/2)+mod(size(normBoundary,1)+1,2):end,:); % Second half of points
    meanDistanceDiagonals(i) = mean(sqrt(sum(((a+x)'.^2),1)));
    symmetryFitness(i) = 1./(1 + meanDistanceDiagonals(i));
end

rawFitness = meanDistanceDiagonals;
fitness = symmetryFitness';

% Limit fitness between 0 and 1
fitness(fitness>1) = 1;
fitness(fitness<0) = 0;

end

%------------- END CODE --------------