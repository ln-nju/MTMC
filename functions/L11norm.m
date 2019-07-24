function [Xnorm] = L11norm(X)
Xnorm = sum(sum(abs(X)));
