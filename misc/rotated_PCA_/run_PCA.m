%
%
%
%
%

%% load data
data = load('rotated_PCA_input_data.mat');
X = data.X;
[n, p] = size(X);
mu_X = mean(X, 1);
Xp = X - mu_X;

%% run PCA
%  centers the data automatically and uses the SVD formulation
%
%  coeff:  p x p  principal directions/axes
%  score:  n x p  principal components (PCs)
%  latent: p x 1  eigenvalues of data covariance matrix (squared correlation coeffs; and variances of the PCs themselves)
[coeff, score, latent] = pca(X);

%loading = diag(latent) * coeff;  % incorrect; could use bsxfun instead
loading = coeff * diag(sqrt(latent));

% normalize the PCs
%score_N = score * diag(1./sqrt(latent));
score_N = score * diag(latent.^-0.5);

tol = 1e-6;
assert( sum(abs(std(score, 1) - sqrt(latent'))) < 1.0 )  % variance of PC cols == eigvals; not close to the tol, but < 1...
assert( sum(abs(vecnorm(coeff) - ones(1, p))) < tol )  % principal axes are unit vectors (in both col and row direction => orthogonal matrix) 
assert( sum(sum(abs(corr(score) - eye(p)))) < tol )    % PCs are uncorrelated
assert( sum(sum(abs(score - Xp*coeff))) < tol )        % check that PCs are formed by X V; though pca() computes it as U S
% ^ these columns are orthogonal
%   this means they are *not* the loading vectors, contrary to what Matlab says
%   by default vecnorm computes along axis 1 (rows) with 2-norm

%% varimax rotation
%  by default, rotatefactors() does varimax with row normalization
n_rot = 6;
reltol_rot = 1.0e-8;
maxit_rot = 2000;

X_hat0 = score_N(:,1:n_rot) * loading(:,1:n_rot)';
X_hat0_2 = score(:,1:n_rot) * coeff(:,1:n_rot)';  % same

% use principal axes as input
[coeff_rot, T_coeff] = rotatefactors(coeff(:,1:n_rot),...
    'Method', 'varimax', 'Normalize', 'off', 'RelTol', reltol_rot, 'Maxit', maxit_rot);

% use Kaiser row-normalized principal axes as input
% let rotatefactors() do the normalizing
[coeff_rot_KN, T_coeff_KN] = rotatefactors(coeff(:,1:n_rot),...
    'Method', 'varimax', 'Normalize', 'on', 'RelTol', reltol_rot, 'Maxit', maxit_rot);


% ----
% note that it is the loadings and normalized PCs that are supposed to be rotated
% (not the eigenvectors or raw PCs)
% but we can recover the latter using the former

% use raw loadings as input
[loading_rot, T_loading] = rotatefactors(loading(:,1:n_rot),...
    'Method', 'varimax', 'Normalize', 'off', 'RelTol', reltol_rot, 'Maxit', maxit_rot);
latent_rot_loading = vecnorm(loading_rot).^2;
coeff_rot_loading = loading_rot * diag(latent_rot_loading.^-0.5);
assert( sum(abs(vecnorm(coeff_rot_loading) - ones(1, n_rot))) < tol )  % new axes still unit vectors
% but not orthogonal!
assert( abs(sum(sum(coeff_rot_loading' * coeff_rot_loading - eye(n_rot)))) > tol  )
score_N_rot_loading = score_N(:,1:n_rot) * T_loading;  % rotate the normalized PCs
score_rot_loading = score_N_rot_loading * diag(latent_rot_loading.^0.5);  % recover rotated raw PCs
assert( sum(sum(abs(corr(score_rot_loading) - eye(n_rot)))) < tol )    % new components are still uncorrelated
assert( sum(sum(abs(X_hat0 - score_N_rot_loading*loading_rot'))) < tol )  % same reconstruction
assert( var(latent_rot_loading) > var(latent) )  % greater spread of explained variance among components


% use Kaiser row-normalized loadings as input
% though could also use the 'Normalize' option
norms = vecnorm(loading, 2, 2);
loading_KN = diag(1./norms) * loading;
[loading_KN_rot, T_loading_KN] = rotatefactors(loading_KN(:,1:n_rot),...
    'Method', 'varimax', 'Normalize', 'off', 'RelTol', reltol_rot, 'Maxit', maxit_rot);
loading_rot_KN = diag(norms) * loading_KN_rot;

%loading_rot2 - rotatefactors(loading(:,1:n_rot), 'RelTol', reltol_rot, 'Maxit', maxit_rot)  % < should be small


%% save PCA and rotation matrix results
save run_PCA_out.mat coeff score latent loading T_coeff T_coeff_KN T_loading T_loading_KN

