%% FUNCTION MTMC
%  Use FISTA algorithm to solve the problem
%% INPUT
%   X: {n * d } * t - input matrix
%   Y: {n * 1} * t - output matrix
%   rho1: {C} - group sparsity regularization parameter
%   rho2: {C} - elementwise sparsity regularization parameter
%
%% OUTPUT
%   W: model: C * (d+1) * t    
%   P: group sparsity structure (joint feature selection)
%   Q: elementwise sparsity component
%   funcVal: function (objective) value vector.
%   lossVal: loss value for every iteration.

function [W, P, Q, funcVal, lossVal] = MTMC(X, Y, rho1, rho2, opts, class_num)

if length(rho1)==1 
    rho1 = ones(1, class_num) * rho1;
end

if length(rho2)==1 
    rho2 = ones(1, class_num) * rho2;
end

if nargin < 4
    error('\n Inputs: X, Y, rho1, rho2, should be specified!\n');
end

if nargin < 5
    opts = [];
end

task_num = length(X);
for i = 1 : task_num
    n = size(X{i}, 1);
    intercept = ones(n, 1); %intercept term
    X{i} = cat(2, intercept, X{i});
end

dimension = size(X{1}, 2); % d + 1
funcVal = [];
lossVal =[];

if opts.init == 2
    P0 = zeros( class_num, dimension, task_num);
    Q0 = zeros(class_num, dimension, task_num);
elseif opts.init == 0
    P0 = randn(class_num, dimension, task_num);
    Q0 = randn(class_num, dimension, task_num);
elseif opts.init == 1
    if isfield(opts, 'P0')
        P0 = opts.P0;
        if(nnz(size(P0) - [class_num, dimension, task_num]))
            error('\n Check the input .P0');
        end
    else 
        error('\n check opt.init');
    end
    
    if isfield(opts, 'Q0')
        Q0 = opts.Q0;
        if(nnz(size(Q0) - [class_num, dimension, task_num]))
            error('\n Check the input .Q0');
        end
    else 
        error('\n check opt.init');
    end
end

Pz = P0;  
Qz = Q0;
Pz_old = P0;  
Qz_old = Q0;

t = 1;
t_old = 0;
iter = 0; 
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) / t;
    
    Ps = (1 + alpha) * Pz - alpha * Pz_old;
    Qs = (1 + alpha) * Qz - alpha * Qz_old;
    
    [gWs, Fs] = gradVal_eval(Ps + Qs);
    
    while true
        for c = 1 : class_num
            Ptmp = proximalL21norm(squeeze(Ps(c,:,:)) - squeeze(gWs(c,:,:)) / gamma, rho1(c) / gamma);
            Qtmp = proximalL11norm(squeeze(Qs(c,:,:)) - squeeze(gWs(c,:,:))/gamma, rho2(c) / gamma);
            Pzp(c, :, :) = permute(Ptmp, [3,1,2]);
            Qzp(c, :, :) = permute(Qtmp, [3,1,2]);
        end 
        Fzp = funVal_eval (Pzp + Qzp);
        
        delta_Pzp = Pzp - Ps;
        delta_Qzp = Qzp - Qs;
        nm_delta_Pzp = sum(sum(sum(delta_Pzp .^ 2)));   
        nm_delta_Qzp = sum(sum(sum(delta_Qzp .^ 2)));
      
        r_sum = (nm_delta_Pzp + nm_delta_Qzp) / (2 * class_num); % mean value
        
        Fzp_gamma  = Fs + sum(sum( sum( ( delta_Pzp + delta_Qzp ) .* gWs) ))...
            + gamma / 2 * (nm_delta_Pzp + nm_delta_Qzp);
        
        if r_sum <= 1e-20
            break;
        end
       
        if Fzp <= Fzp_gamma
            break;
        else
            gamma = gamma * gamma_inc;
        end     
    end
    
    Pz_old = Pz;
    Qz_old = Qz;
    Pz = Pzp;
    Qz = Qzp;
    
    L21sum = 0;
    L11sum = 0;
    for c = 1 : class_num
        L21sum =L21sum + rho1(c) * L21norm(squeeze(Pzp(c,:,:)));
        L11sum = L11sum + rho2(c) * L11norm(squeeze(Qzp(c,:,:)));
    end
    funcVal = cat(1, funcVal, Fzp + L21sum + L11sum);
   
    lossVal = cat(1, lossVal, Fzp);
    
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))                   
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);    
end

P=Pzp;
Q=Qzp;
W = Pzp+Qzp;
   
    function [grad_W, funcVal] = gradVal_eval(W)
        grad_W = zeros(class_num, dimension, task_num);
        lossValVect = zeros (1 , task_num);
        for k = 1:task_num
                [ grad_W(:, :, k), lossValVect(:, k)] = Softmax_Loss_grad( W(:, :, k), X{k}, Y{k});
        end
        funcVal = sum(lossValVect);
    end

    function[funcVal] = funVal_eval (W)
        funcVal = 0;
        for j = 1: task_num
            funcVal = funcVal + Softmax_Loss ( W(:, :, j), X{j}, Y{j} );
        end
    end
end
