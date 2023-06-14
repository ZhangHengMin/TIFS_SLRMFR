function [R] = CoordinateDescent(M, k, MaxIter)
%Compute the row weight matrix R by coordinate descent
% Requirement:
%   1. only for uniform sampling model

n1 = size(M, 1);
R  = ones(n1, 1);


% --------------------------- Parameters ---------------------------%
isSparse = true; % true if M is sparse
rho      = k;    % should be tuned


% --------------------------- Default ---------------------------%
if nargin < 3
    MaxIter = k^2;
end

% --------------------------- Iterations ---------------------------%
for iter = 1: MaxIter
    % Compute the leverage scores of M_k
    if isSparse
        [U, ~, ~] = svds(M, k);
    else
        [U, ~, ~] = svd(M, 'econ');
        U = U(:, 1:k);
    end
    hmu = sum(U.^2, 2);
    
    % Find the max leverage score of M
    [hmu_max, idx] = max(hmu);
    
    if mod(iter, 100) == 0
        disp(['Iter: ', int2str(iter), ', Max leverage score = ', num2str(hmu_max)]);
    end
    
    % Compute the weight
    if hmu_max > 1 - 1 / rho
        gamma = ComputeGamma(hmu_max - 0.5/rho, 1/rho);
    else
        gamma = ComputeGamma(hmu_max, 2*k/n1);
    end
    if gamma > 0.995
        gamma = 0.995;
    end
    weight = sqrt(1-gamma);
    R(idx) = R(idx) * weight;
    
    % Reweigh the rows of M
    M(idx, :) = M(idx, :) * weight;
end



end