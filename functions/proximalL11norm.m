function [X] = proximalL11norm(D, tau)
X = sign(D).*max(0,abs(D)-tau);
