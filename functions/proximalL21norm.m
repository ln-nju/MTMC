function [X] = proximalL21norm(D, tau)
X = repmat(max(0, 1 - tau./sqrt(sum(D.^2,2))),1,size(D,2)).*D;