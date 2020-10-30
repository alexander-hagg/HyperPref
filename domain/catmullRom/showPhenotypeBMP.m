function figHandle = showPhenotypeBMP(genomes,d,varargin)
%showPhenotype - Either show an example phenotype, or, when given, show
%                multiple phenotypes that are positioned on predefined placement positions.
%                Yes, this visualization script does too many things at the same time.
%
% Syntax:   showPhenotype(genomes,d,varargin)
%
% Inputs:
%    figHandle      - [1] - figure handle
%    d              - struct - Domain description struct
%
% Optional Inputs:
%    varargin{1}    - [NxM] - genomes
%    varargin{2}    - [Nx2] - placement
%    varargin{3}    - [Nx1] - selection labels (1 or 2)
%
%
% Author: Alexander Hagg
% Bonn-Rhein-Sieg University of Applied Sciences (HBRS)
% email: alexander.hagg@h-brs.de
% Aug 2019; Last revision: 15-Aug-2019
%
%------------- BEGIN CODE --------------
nShapes = size(genomes,1);
xPos = 0:ceil(sqrt(nShapes))-1; [X,Y] = ndgrid(xPos,xPos);
placement = 1 + [X(:) Y(:)]*d.resolution;
if nargin>2
    if ~isempty(varargin{1})
        figHandle = varargin{1};
    else
        figHandle = figure;figHandle = showPhenotype(genomes,d,varargin)
    end
else
    figHandle = figure;
end
if nargin>3
    if ~isempty(varargin{2})
        placement = varargin{2};
    end
end

if nargin>4
    clrs = varargin{3};
else
    clrs = [0 0 0];
end

booleanMaps = d.getPhenotype(genomes);

placeRange = (max(placement(:))-min(placement(:)));
bitmap = logical(zeros(placeRange,placeRange));
for i=1:nShapes
    if ~isempty(booleanMaps{i})
        pl = placement(i,:);
        bitmap(pl(1):pl(1)+d.resolution-1,pl(2):pl(2)+d.resolution-1) = booleanMaps{i};
        %bitmap((((pl(1)-1))*placeRange)*2 + [1:d.resolution],((pl(2)-1)*placeRange)*2 + [1:d.resolution]) = booleanMaps{i};
        
    end
end

bitmap = bitmap'; bitmap = flipud(bitmap);
figure(figHandle);
bla = imshow(bitmap);
%axis([-d.resolution/2 bitmapRes+d.resolution/2 -d.resolution/2 bitmapRes+d.resolution/2]);

drawnow;

end

