function [R] = CoordinateDescentLowRank(M, k, r, MaxIter)
%Compute the row weight matrix R by coordinate descent
% Requirements:
%   1. rank(M) = r, which is far less than n2
%   2. only for uniform sampling model

n1 = size(M, 1);
R = ones(n1, 1);

% --------------------------- Parameters ---------------------------%
rho = k; % should be tuned



% --------------------------- Default ---------------------------%
if nargin < 3
    r = rank(full(M));
end
if nargin < 4
    MaxIter = k^2;
end




% --------------------------- Replace M by U ---------------------------%
% [M, ~, ~] = svds(M, r); % sometimes slower
[M, ~, ~] = svd(full(M), 'econ');
M = M(:, 1: r);
U = M(:, 1:k);
hmu = sum(U.^2, 2);


% --------------------------- Iterations ---------------------------%
for iter = 1: MaxIter
    % Find the max leverage score of M
    [hmu_max, idx] = max(hmu);
    
    if mod(iter, 100) == 0
        disp(['Iter: ', int2str(iter), ',    Max leverage score = ', num2str(hmu_max)]);
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
    [U, ~, ~] = svd(M, 'econ');
    U = U(:, 1:k);
    hmu = sum(U.^2, 2);
end



end