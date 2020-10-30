clear;
%% Configuration
addpath(genpath(pwd))                           % Set path to all modules
DOF = 16;DOMAIN = 'catmullRom';                 % Degrees of freedom, Catmull-Rom spline domain
d = domain(DOF);                                % Domain configuration

% Create shapes variations
shapeParams = [1 0.5 1 0.5 1 0.5 1 0.5 zeros(1,8)];
%shapeParams = [0 0.1 1 0.5 2 0.0 1 0.5 zeros(1,8)];
%shapeParams = [0 0.0 1 0.5 0.2 0.0 0 0.5 zeros(1,8)];
intermediate = getPhenotypeFFD(shapeParams,d.base);
[fitness,~,~] = fitfun(shapeParams,d)
[~,booleanMap] = getPhenotypeBoolean(intermediate);
[fitness,~,~] = fitfun(booleanMap,d)


B = bwboundaries(booleanMap{1},'noholes');
boundary = B{1};
boundImg = zeros(size(booleanMap{1}));
for pix=1:length(B{1}(:,1))
    boundImg(B{1}(pix,1),B{1}(pix,2)) = 1;
end
% Show centroid
centroid = mean(boundary);
boundImg(ceil(centroid(1)),ceil(centroid(2))) = 2;

%normalize boundary coordinates for fitness function
normBoundary = mapminmax(boundary',-1,1)';
if ~mod(size(normBoundary,1),2)==0; normBoundary(end,:) = []; end % Make even number # points
a = normBoundary(1:ceil(end/2),:); % Take first half of points
x = normBoundary(ceil(end/2)+mod(size(normBoundary,1)+1,2):end,:); % Second half of points
meanDistanceDiagonals = mean(sqrt(sum(((a+x)'.^2),1)));
symmetryFitness = 1./(1 + meanDistanceDiagonals);

fig(1) = figure(1); hold off; ax = gca;
imagesc(ax,0.5*booleanMap{1}+boundImg);
hold on;
title(['Symmetry Fitness: ' num2str(symmetryFitness)]);

%%
fig(2) = figure(2); hold off; ax = gca;
axImg = zeros(size(booleanMap{1}));
a = boundary(1:ceil(end/2),:); % Take first half of points
x = boundary(ceil(end/2)+mod(size(normBoundary,1)+1,2):end,:); % Second half of points

for pix=1:length(a(:,1))
    axImg(a(pix,1),a(pix,2)) = pix/length(a(:,1));
end
for pix=1:length(x(:,1))
    axImg(x(pix,1),x(pix,2)) = pix/length(a(:,1));
end
imagesc(ax,axImg);


