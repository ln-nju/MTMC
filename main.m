clear;
clc;
close;

addpath('./functions/');  % load function

T = 2;  % task number
cNum = 11;  % class number

% data preparation
X = cell(T, 1);
Y = cell(T, 1);  % true label
for i = 1 : T
    filename = ['.\data\riverS_' num2str(i) '.csv'];
    data = csvread(filename);
    [samplenum, col] = size(data);
    r=randperm(size(data,1));   
    X{i} = data(:, 1:col -1);
    Y{i} = data(:, col); 
end

% normalization
for i =1: T
    X{i} = mapminmax((X{i}+0.000000000000001)', 0, 1)';
end

% optimization options
opts.init = 0;  
opts.tFlag = 1; 
opts.tol = 10^-4;
opts.maxIter = 200; 

% lambda range
lambda1_range = 1:-0.0001:0.001; 
lambda2_range = 1:-0.0001:0.001;

%nested cross validation 
out_cv_fold =10;  % out cross validation for final results
in_cv_fold = 5;  % in cross validation for best parameters
perform_measure = cell(T, 1);  % save the results
Indices = cell(T, 1);
preFile = 'preLabel.txt';
trueFile = 'trueLabel.txt';
% K fold
for t = 1 : T
    Indices{t} = crossvalind('Kfold', size(X{t},1), out_cv_fold );
end

pfile = fopen(preFile, 'a');
tfile = fopen(trueFile, 'a');
for i = 1 : out_cv_fold   
    fprintf(pfile, '%s\r\n',['*************************fold ' num2str(i) '*************************']);
    fprintf(tfile, '%s\r\n',['*************************fold ' num2str(i) '*************************']);
    Ypre = cell(T, 1);%Ô¤²â½á¹û
    
    fprintf('current fold: %d\n',i);    
    Xtr = cell(T, 1);
    Ytr = cell(T, 1);
    Xte = cell(T, 1);
    Yte = cell(T, 1);

    for t = 1: T
        te_idx = (Indices{t} == i);
        tr_idx = ~te_idx;
        % split data for cross validation
        Ytr{t} = Y{t}(tr_idx, :);  
        Xtr{t} = X{t}(tr_idx, :);
        Yte{t} = Y{t}(te_idx, :); 
        Xte{t} = X{t}(te_idx, :); 
        
        fprintf(tfile,'%s\r\n',['Task ' num2str(t) ':']);
        fprintf(tfile,'%d\t',Yte{t});
        fprintf(tfile,'\r\n');
    end
    
    % inner CV, get the best lambda1 and lambda2
    fprintf('inner CV started \n');
    [best_lambda1, best_lambda2, accuracy_mat] = CrossValidation( Xtr, Ytr, ...
            'MTMC', opts, lambda1_range,lambda2_range, in_cv_fold, ...
            'eval_f1_allTask', cNum);   
    % train
    % get the best opts by fixing lambda1 and lambda2
    [W, P, Q, funcVal, lossVal] = MTMC(Xtr, Ytr, best_lambda1, best_lambda2, opts, cNum);
    opts2=opts;
    opts2.init=1;
    opts2.P0=P;
    opts2.Q0=Q;
    opts2.tol = 10^-4;        
    [W2, P2, Q2, funcVal2, lossVal2] = MTMC(Xtr, Ytr, best_lambda1, best_lambda2, opts2, cNum);
    
    % Prediction and save for each task
    for t = 1 : T
        Ypre{t} = Softmax_Prediction(W2(:,:,t), Xte{t});
        fprintf(pfile,'%s\r\n',['Task ' num2str(t) ':']);
        fprintf(pfile,'%d\t',Ypre{t});
        fprintf(pfile,'\r\n');
    end
      
    % Record performance measure
    measure= Precision_Recall_F1_eachT(Ypre, Yte, cNum);
    for k = 1 : T
        perform_measure{k}(i,:) = measure(k,:); 
    end
end 

fclose(pfile);
fclose(tfile);

final_result = cell(T,1);
for k = 1 : T
    final_result{k} = [mean(perform_measure{k}); std(perform_measure{k})];
end
format long
celldisp(final_result);