function [Y_Pre] = Softmax_Prediction(W, X)
n = size(X, 1);
intercept = ones(n, 1);  
X = cat(2, intercept, X);

P = zeros(n, size(W, 1));  
for i = 1 : n
    Xw = X(i, :) * W';
    Xw = Xw - max(Xw);
    P(i, :) = exp(Xw) / sum( exp(Xw));
end

[~, Y_Pre] = max(P, [], 2);
end