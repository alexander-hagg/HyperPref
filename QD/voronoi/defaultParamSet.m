function p = defaultParamSet(varargin)
% defaultParamSet - loads default parameters for QD algorithm with Voronoi
% feature space
% (here: MAP-Elites with Voronoi archive)
%
% Syntax:  p = defaultParamSet(varargin)
%
% Outputs:
%   p      - struct - QD configuration struct
%
% Author: Alexander Hagg
% Bonn-Rhein-Sieg University of Applied Sciences (HBRS)
% email: alexander.hagg@h-brs.de
% Nov 2019; Last revision: 13-Nov-2019
%
%------------- BEGIN CODE --------------

p.nGens                     = 2^11;          % number of generations
p.nChildren                 = 2^5;          % number of children per generation
p.mutSigma                  = 0.1;          % mutation drawn from Gaussian distribution with this \sigma
p.maxBins                   = 256;    

p.extraMapValues            = {};%{'uncertainty'};

end


%------------- END OF CODE --------------