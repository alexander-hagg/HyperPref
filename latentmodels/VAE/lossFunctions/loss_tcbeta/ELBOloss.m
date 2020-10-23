function [elbo,reconstructionLoss,KL] = ELBOloss(x, xPred, zMean, zLogvar, z, epoch, numEpochs)
% https://github.com/YannDubs/disentangling-vae
alpha = 1; beta = 1; gamma = 1;
squares = 0.5*(xPred-x).^2;
reconstructionLoss  = sum(squares, [1,2,3]);


% Gather data

zMean = double(gather(extractdata(zMean)))';
zLogvar = double(gather(extractdata(zLogvar)))';
if isa(z,'dlarray')
    z = gather(extractdata(z));
    z = squeeze(z);
end
z = double(z)';

%%
% old KL = -.5 * sum(1 + zLogvar - zMean.^2 - exp(zLogvar), 1);


[log_pz, log_qz, log_prod_qzi, log_q_zCx] = get_log_pz_qz_prodzi_qzCx(z, zMean, zLogvar);
%%%% I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
mi_loss = -mean(log_q_zCx - log_qz);
%%%% TC[z] = KL[q(z)||\prod_i z_i]
tc_loss = -mean(log_qz - log_prod_qzi);
%%%% dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
dw_kl_loss = -mean(log_prod_qzi - log_pz);


%%
annealingScalar = linear_annealing(0, 1, epoch, numEpochs);
%%%% total loss
%%%% from py repo - loss = rec_loss + (self.alpha * mi_loss + self.beta * tc_loss + anneal_reg * self.gamma * dw_kl_loss)
KL = alpha .* mi_loss + beta .* tc_loss + annealingScalar .* gamma .* dw_kl_loss;
elbo = mean(reconstructionLoss + KL);

% For output
reconstructionLoss = mean(reconstructionLoss);
KL = mean(KL);
KL = gpuArray(dlarray(KL));
    
function anneal = linear_annealing(init, fin, step, maxSteps)

if step == 0
    anneal = fin;
    return;
end

if fin <= init
    disp('Annealing: final value is not larger than initial value!');
    anneal = fin;
    return;
end

delta = fin - init;
anneal = init + step*(delta/maxSteps);

function [log_pz, log_qz, log_prod_qzi, log_q_zCx] = get_log_pz_qz_prodzi_qzCx(z, zMean, zLogvar)
% NOTES
% latent_sample = z
% latent_dist = zMean and zLogvar

%%%% calculate log q(z|x)
% log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)
log_density = log_density_gaussian(z,zMean,zLogvar);   
log_q_zCx = sum(log_density,2);

%%%% calculate log p(z)
%%%% mean and log var is 0
% zeros = torch.zeros_like(latent_sample)
zeroMat = zeros(size(z));
% log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)
log_density = log_density_gaussian(z,zeroMat,zeroMat);   
log_pz = sum(log_density,2);

% mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)
%    batch_size, dim = x.shape
%    x = x.view(batch_size, 1, dim)
zMat = reshape(z,size(z,1),1,size(z,2));
%    mu = mu.view(1, batch_size, dim)
meanMat = reshape(zMean,1,size(z,1),size(z,2));
%    logvar = logvar.view(1, batch_size, dim)
logvarMat = reshape(zLogvar,1,size(z,1),size(z,2));

%    log_density = log_density_gaussian(x, mu, logvar)
mat_log_qz = log_density_gaussian(zMat,meanMat,logvarMat);
mat_log_qz = double(mat_log_qz);
% log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
% Returns the log of summed exponentials of each row of the input tensor in the given dimension dim
% IS THIS CORRECT????
tempSum = squeeze(sum(mat_log_qz,3));
tempSum = sum(exp(tempSum),2);
%log_qz = log(sum(tempSum(:)))
log_qz = log(tempSum);

% log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)
tempSum = squeeze(mat_log_qz);
tempSum = squeeze(sum(exp(tempSum),2));
%log_qz = log(sum(tempSum(:)))
tempSum = log(tempSum);
tempSum = squeeze(sum(tempSum,2));
log_prod_qzi = tempSum;

function log_density = log_density_gaussian(sampleZ,meanZ,logvar)
log_density = lognpdf(sampleZ,meanZ,exp(logvar));

%normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
%inv_var = torch.exp(-logvar)
%log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
%normalization = - 0.5 * (log(2 * pi) + logvar);
%inv_var = exp(-logvar);
%log_density = normalization - 0.5 * ((sampleZ - meanZ).^2 .* inv_var);    
