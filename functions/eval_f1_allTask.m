function [f1List, F1,acc] = eval_f1_allTask(Y, X, W)
  task_num = length(Y);
  Y_pre = cell(task_num, 1);
  cNum = size(W, 1);
  for n =1 : task_num
      Y_pre{n} = Softmax_Prediction(W(:,:,n), X{n});
  end
  pList = zeros(1,cNum); 
  rList = zeros(1,cNum) ; 
  acc_num = zeros(1,cNum); 
  sample_num = 0; 
  for t = 1 : task_num
      sample_num = sample_num + length(Y{t}); 
      for i = 1 : cNum
          pList(1,i) = pList(1,i) + nnz(Y_pre{t} == i);  
          rList(1,i) = rList(1,i) + nnz(Y{t} == i); 
          acc_num(1,i) = acc_num(1,i) + nnz( (Y_pre{t} == Y{t}) & (Y{t} == i)); 
      end
   end
      
      rList1 = rList; 
      pList = acc_num ./ (pList + 10^-10);
      rList = acc_num ./ (rList + 10^-10);
      pr_sum = pList + rList;
      f1List = (2 * pList .* rList) ./ (pr_sum + 10^-10); 
      acc = sum(acc_num) / sample_num;
      F1 = sum(rList1 .* f1List) / sample_num;
end