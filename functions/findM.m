function [M] = findM(tau, row)
   m = length(row); 
   M = 1;
   while M < m
       sum1 = sum(row(1: M+1) - row(M+1));
       sum2 = sum((row(1:M) - row(M)));
       if sum1 >= row(M+1) && sum2 * tau < row(M)
            break;
       else
            M = M +1;
       end
   end
end
  