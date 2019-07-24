function f1 = eval_MTL_f1(Y, X, W)
    task_num = length(Y);
    Y_pre = cell(task_num, 1);
    cNum = size(W, 1);
    for n =1 : task_num
        Y_pre{n} = Softmax_Prediction(W(:,:,n), X{n});
    end
    
    P_R_F_eTask = Precision_Recall_F1_eachT(Y_pre, Y, cNum);
    eF1 = P_R_F_eTask(:,3);
    f1 = sum(eF1) / task_num;
end