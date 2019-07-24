function [funcVal] = Softmax_Loss(W, X, Y)
n = size(X, 1);
Y_extend = zeros(n, size(W, 1));
for i = 1:n
    Y_extend(i, Y(i)) = 1;
end
% Compute the probability matrix
P = zeros(n, size(Y_extend, 2));
for i = 1 : n
    Xw = X(i, :) * W';
    Xw = Xw - max(Xw); 
    P(i, :) = exp(Xw) / sum (exp(Xw));
end 
funcVal =  -1/n * (Y_extend(:)' * log(P(:)));
end