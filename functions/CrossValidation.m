function [ best_lambda1 best_lambda2 perform_mat] = CrossValidation...
    ( X, Y, obj_func_str, obj_func_opts, lambda1_range, lambda2_range, cv_fold, eval_func_str, cNum)

eval_func = str2func(eval_func_str);
obj_func  = str2func(obj_func_str);

best_lambda1 = zeros(1, cNum);
best_lambda2 = zeros(1, cNum);

task_num = length(X);

perform_mat = zeros(length(lambda1_range),length(lambda2_range)*cNum);

% seperate data for cross validation
Indices = cell(task_num, 1);
for k = 1 : task_num
    Indices{k} = crossvalind('Kfold', size(X{k},1), cv_fold);
end

% begin cross validation
fprintf('[')
for cv_idx = 1: cv_fold
    fprintf('.')
    
    % buid cross validation data splittings for each task.
    cv_Xtr = cell(task_num, 1);
    cv_Ytr = cell(task_num, 1);
    cv_Xte = cell(task_num, 1);
    cv_Yte = cell(task_num, 1);
    
    for t = 1: task_num
        te_idx = (Indices{t} == cv_idx);
        tr_idx = ~te_idx;
        cv_Xtr{t} = X{t}(tr_idx, :);
        cv_Ytr{t} = Y{t}(tr_idx, :);
        cv_Xte{t} = X{t}(te_idx, :);
        cv_Yte{t} = Y{t}(te_idx, :);
    end
    perform = cell (length(lambda1_range), length(lambda2_range));
  
    parfor k = 1 : length(lambda1_range) * length(lambda2_range)
        i=mod(k-1,length(lambda1_range))+1;
        ii=floor((k-1)/length(lambda1_range))+1;
        opts=obj_func_opts;
        opts.init=0;
        if(isfield(opts, 'P0'))
            opts = rmfield(opts, 'P0');
        end
        if(isfield(opts, 'Q0'))
            opts = rmfield(opts, 'Q0');
        end
        
        [W, P, Q] = obj_func(cv_Xtr, cv_Ytr, lambda1_range(i), lambda2_range(ii), opts, cNum);
        opts.init=1;
        opts.P0=P;
        opts.Q0=Q;
        perform{k} = eval_func(cv_Yte, cv_Xte, W);
    end
    perform_mat = perform_mat+cell2mat(perform);
end
           
perform_mat = permute(reshape(perform_mat,length(lambda1_range),cNum,length(lambda2_range)),[1,3,2]);
 
perform_mat = perform_mat./cv_fold;
fprintf(']\n')
    for c = 1: cNum
        M = perform_mat(:,:,c);
        [lambda1_idx,  lambda2_idx] = find(M==max(max(M)));
        best_lambda1(c)=lambda1_range(lambda1_idx(end));
        best_lambda2(c)=lambda2_range(lambda2_idx(end));
    end
end

