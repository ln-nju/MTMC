function [P_R_F_eTask] = Precision_Recall_F1_eachT(Y_pre, Y, cNum)
  task_num = length(Y);
  P_R_F_eTask = zeros(task_num,3);
  for t = 1 : task_num
      pList = zeros(1,cNum); 
      rList = zeros(1,cNum) ; 
      acc_num = zeros(1,cNum); 
      sample_num = length(Y{t});
      for i = 1 : cNum
          pList(1,i) = nnz(Y_pre{t} == i);  
          rList(1,i) = nnz(Y{t} == i); 
          acc_num(1,i) = nnz( (Y_pre{t} == Y{t}) & (Y{t} == i)); 
      end
     
      rList1 = rList; 
      pList = acc_num ./ (pList + 10^-10);
      rList = acc_num ./ (rList + 10^-10);
      pr_sum = pList + rList;
      disp(['Task', num2str(t)]);
      f1List = (2 * pList .* rList) ./ (pr_sum + 10^-10)
      
      P_Taskt= sum(rList1 .* pList) / sample_num;
      R_Taskt = sum(rList1 .* rList) / sample_num;
      F1_Taskt = sum(rList1 .* f1List) / sample_num;
      P_R_F_eTask(t, :) = [P_Taskt, R_Taskt, F1_Taskt];
  end
end