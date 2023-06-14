function [gamma] = ComputeGamma(lev0, lev1)

gamma = 1 - lev1 ./ lev0;
gamma = gamma ./ (1 - lev1);

end