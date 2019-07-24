function [Xnorm] = L21norm(X)
Xnorm = sum(sqrt(sum(X.^2,2)));
