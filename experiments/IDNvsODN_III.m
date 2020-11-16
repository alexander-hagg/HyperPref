clear;
%% Configuration
addpath(genpath(pwd))                           % Set path to all modules
DOF = 16;                                       % Degrees of freedom, Catmull-Rom spline domain
d = domain(DOF);                                % Domain configuration
FITNESSFUNCTION = 'bmpSymmetry'; rmpath(genpath('domain/catmullRom/fitnessFunctions')); addpath(genpath(['domain/catmullRom/fitnessFunctions/' FITNESSFUNCTION]));
numInitSamples = 512;
baseFilename = ['catmullRom_IIIb'];

latentDOFs = [8 16 32];
dimRange = [-3, 3];

ALGORITHM = 'voronoi'; rmpath('QD/mapelites'); rmpath('QD/voronoi'); addpath(['QD/' ALGORITHM]);

%% STARTED 02:44
%% Run experiments, varying latent DOF
for replicate=3:10
    fileName = [baseFilename '_replicate_' int2str(replicate)];
    clear map config results;
    load([fileName '.mat']);
    map(2,:) = map(1,:);
    results(2,:) = results(1,:);
    config(2,:) = config(1,:);
    latentDomain(replicate,2,:) = latentDomain(replicate,1,:);

    latentDOFs = [8 16 32];
    for rep=1:2:length(latentDOFs)
        %% Initialize solution set using space filling Sobol sequence in genetic space
        sobSequence = scramble(sobolset(d.dof,'Skip',1e3),'MatousekAffineOwen');  sobPoint = (replicate-1)*numInitSamples+1;
        initSamples{rep,1} = range(d.ranges').*sobSequence(sobPoint:(sobPoint+numInitSamples)-1,:)+d.ranges(:,1)';
        [~,initPhenotypes{rep,1}] = fitfun(initSamples{rep,1},d);
        
        p = defaultParamSet(latentDOFs(rep));                            % Base Quality Diversity (QD) configuration (MAP-Elites)
        
        % Run POEM on parameter space with latent space niching
        m = cfgLatentModel('data/workdir',d.resolution,latentDOFs(rep));                    % VAE configuration
        pm = poemParamSet(p,m);                                             % Configure POEM ("Phenotypic niching based Optimization by Evolving a Manifold")
        pm.map.numInitSamples = numInitSamples;
        pm.map.maxBins = numInitSamples;
        pm.categorize = @(geno,pheno,p,d) predictFeatures(pheno,p.model);   % Anonymous function ptr to phenotypic categorization function (= VAE)
        
        [map{rep,1}, config{rep,1}, results{rep,1}] = poem(initSamples{rep,1},pm,d);
        %save([fileName '.mat'],'map','config','results','pm','p','m','initSamples','initPhenotypes','latentDOFs','dimRange');
        disp('Finished POEM on parameter space (III)');
        
        % Run POEM on latent space with latent space niching
        latentDomain{replicate,rep,2} = d; latentDomain{replicate,rep,2}.dof = latentDOFs(rep); latentDomain{replicate,rep,2}.ranges = repmat(dimRange,latentDOFs(rep),1);
        initSamples{rep,2} = getPrediction(initPhenotypes{rep,1}, results{rep,1}.models(1));
        latentDomain{replicate,rep,2}.getPhenotype = @(latentCoords)sampleVAE(latentCoords,results{rep,1}.models(1).decoderNet);
        [map{rep,2}, config{rep,2}, results{rep,2}] = poem(initSamples{rep,2},pm,latentDomain{replicate,rep,2},results{rep,1}.models(1));
        %save([fileName '.mat'],'map','config','results','pm','p','m','initSamples','initPhenotypes','latentDOFs','dimRange','latentDomain');
        disp('Finished POEM on latent space (III)');
        
        % Run POEM on parameter space with latent space niching with updated model
        initSamples{rep,3} = reshape(map{rep,1}.genes,[],d.dof);
        initSamples{rep,3} = initSamples{rep,3}(all(~isnan(initSamples{rep,3})'),:);
        [~,initPhenotypes{rep,3}] = fitfun(initSamples{rep,3},d);
        [map{rep,3}, config{rep,3}, results{rep,3}] = poem(initSamples{rep,3},pm,d);
        %save([fileName '.mat'],'map','config','results','pm','p','m','initSamples','initPhenotypes','latentDOFs','dimRange','latentDomain');
        disp('Finished POEM on parameter space (IV)');
        
        
        % Run POEM on latent space with latent space niching with updated model
        initSamples{rep,4} = getPrediction(initPhenotypes{rep,3},results{rep,3}.models(1));
        latentDomain{replicate,rep,4} = latentDomain{replicate,rep,2};
        latentDomain{replicate,rep,4}.getPhenotype = @(latentCoords)sampleVAE(latentCoords,results{rep,3}.models(1).decoderNet);
        [map{rep,4}, config{rep,4}, results{rep,4}] = poem(initSamples{rep,4},pm,latentDomain{replicate,rep,4},results{rep,3}.models(1));
        save([fileName '.mat'],'map','config','results','pm','p','m','initSamples','initPhenotypes','latentDOFs','dimRange','latentDomain');
        disp('Finished POEM on latent space (IV)');
    end
end


