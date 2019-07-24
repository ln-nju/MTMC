function [grad_W, funcVal] = Softmax_Loss_grad(W, X, Y)
n = size(X, 1);
Y_extend = zeros(n, size(W, 1));
for i = 1:n
    Y_extend(i, Y(i)) = 1;
end
% Compute the probability matrix
P = zeros(n, size(W, 1));
for i = 1 : n
    Xw = X(i, :) * W';
    Xw = Xw - max(Xw);
    P(i, :) = exp(Xw) / sum (exp(Xw));
end 
% function value
funcVal =  -1/n * Y_extend(:)' * log(P(:));
% gradient 
r=X'*(Y_extend - P);
grad_W = -1/n * ( X' * (Y_extend - P))'; 
end