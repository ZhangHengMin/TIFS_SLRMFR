function [X, E, iter] = S1DLRR(Tr, Te, opts, rank_est, rho, tol, maxIter)
%
%------------------------------
% Model for sp norm regression model
% min  \sum*||E_i||_{Sp} + lambda/2*(|U|^{2}_{F}+|V|^{2}_{F}),
% s.t., X = U*V', E = Te - Tr*X .
%

[l, n1] = size(Tr);
normT = norm(Tr, 'fro');
[~, n2] = size(Te);
p = opts.row;
q = opts.col;
mu = opts.mu; % tunable
lambda = opts.lambda;
%lambda = sqrt(max(n1, n2));
%
if (~exist('DEBUG','var'))
    DEBUG = 1;
end
if nargin < 7
    maxIter = 1000;
end
if nargin < 6
    tol = 1e-3;
end
if nargin < 5
    rho = 1.1;
end
if nargin < 4
    %rank_est = opts.rank;
    rank_est = round(opts.rank*rank_estimation(Tr));
end
% if nargin < 3
%     lambda = sqrt(max(m, n));
% end

% Parameter set
max_mu = 1e20;


for ii = 1:2
    if ii == 1  
        %M{ii} = randn(size(Tr, ii), rank_est);              % fix
        M{ii} = zeros(size(Tr, ii), rank_est);
        [U, aa1] = qr(Tr'*M{ii}, 0);
    else 
        %M{ii} = randn(size(Te, 3-ii), rank_est);            % fix
        M{ii} = zeros(size(Te, 3-ii), rank_est);
        [V, aa2] = qr(Te'*M{ii}, 0);
    end
end

X  = U * V';
a1 = norm(X, 2);
a2 = norm(X, Inf)/lambda;
Y = X/max(a1,a2);
clear aa1 aa2 a1 a2 M{1} M{2};
E  = sparse(l, n2);
Y0 = ones(l, n2);

% Y  = zeros(n1, n2);
% Y0  = zeros(l, n2);
% X  = zeros(n1, n2);
% rand('state',0);
% V  = zeros(n2, rank_est);
IN_train = inv(eye(n1)+ Tr'*Tr);
% Start main loop
iter = 0;
while iter < maxIter
    iter = iter + 1;
    
    % Update U
    U = (X + Y/mu)*V*inv(V'*V + lambda*eye(rank_est)/mu);
    % Update V
    V = (X + Y/mu)'*U*inv(U'*U + lambda*eye(rank_est)/mu);
    
    % Update X
    X = IN_train*[Tr'*(Te-E-Y0/mu)+(U*V'-Y/mu)];
    
    % Update E
    MediaE = Te - Tr*X - Y0/mu;
    parfor j = 1 : n2
        [LL, SS, TT] = svd(reshape(MediaE(:,j),[p,q]), 'econ');
        sigma = diag(SS);
        svp = length(find(sigma > 1/mu));
        if svp>=1
            sigma = sigma(1:svp) - 1/mu;
        else
            svp = 1;
            sigma = 0;
        end
        EE = LL(:,1:svp)*diag(sigma(1:svp))*TT(:,1:svp)';
        E(:,j) = EE(:);
    end
    % Update E 1/2
%    MediaE = Te - Tr*X - Y0/mu;
%     parfor j = 1 : n2
%         [LL, SS, TT] = svd(reshape(MediaE(:,j),[p,q]), 'econ');
%         sigma = diag(SS);
%         ws = ST12(sigma, 1/mu);
%         EE = LL*ws*TT';
%         E(:,j) = EE(:);
%     end
%     parfor j = 1 : n2
%         [LL, SS, TT] = svd(reshape(MediaE(:,j),[p,q]), 'econ');
%         sigma = diag(SS);
%         ws = ST23(sigma, 1/mu);
%         EE = LL*diag(ws)*TT';
%         E(:,j) = EE(:);
%     end
    
    % Stopping criteria
    dX = X - U*V';
    dE = E + Tr*X - Te;
    TrX = Tr*X;
    stopX = norm(dX, 'fro');
    stopE = norm(dE, 'fro');
%     if DEBUG
%         if iter==1 || mod(iter,10)==0 || stopE<tol
%             disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
%                 ',recErr=' num2str(stopE/normT,'%2.3e')]);
%         end
%     end
    
    if  stopE/normT < tol %&& stopX/normT < tol
        break;
    else
        Y0 = Y0 + mu*dE;
        Y = Y + mu*dX;
        mu = min(max_mu, mu*rho);
    end
    
end

% % The half-thresholding operator
% function w = ST12(temp_v, gamma)
% a = temp_v; ro = gamma;
% b = 2/3*a.*(1+cos(2*pi/3-2/3*acos(ro/4*(a/3).^(-3/2))));
% w = diag((a>(54^(1/3)/4)*(2*ro)^(2/3)).*b);

% % % The twothirds-thresholding operator
% function w = ST23(temp_v, beta)
% temp_p = 2*(3*beta^3)^0.25/3;
% temp_w = abs(temp_v) > temp_p;
% temp_w = sign(temp_v).*temp_w;
% pp = acosh((27*temp_v.^2)/16*beta^(-3/2));
% pp = 2/sqrt(3)*beta^(0.25)*(cosh(pp/3).^(0.5));
% w  = temp_w.*((pp + sqrt(2*abs(temp_v)./pp - pp.^2))/2).^3;

% This Function to Estimate the Rank of the Input Matrix
function d = rank_estimation(X)
	
[n m] = size(X);
epsilon = nnz(X)/sqrt(m*n);
mm = min(100, min(m, n));
S0 = lansvd(X, mm, 'L');

S1 = S0(1:end-1)-S0(2:end);
S1_ = S1./mean(S1(end-10:end));
r1 = 0;
lam = 0.05;
while(r1 <= 0)
    for idx = 1:length(S1_)
        cost(idx) = lam*max(S1_(idx:end)) + idx;
    end
    [v2 i2] = min(cost);
    r1 = 2*max(i2-1);
    lam = lam+0.05;
end
clear cost;

for idx = 1:length(S0)-1
    cost(idx) = (S0(idx+1)+sqrt(idx*epsilon)*S0(1)/epsilon )/S0(idx);
end
[v2 i2] = min(cost);
r2 = 2*max(i2);

d = max([r1 r2]);
