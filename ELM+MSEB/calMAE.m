% Author: Tsien at NUAA, China. 05/09/2015
% Description:
% This function is to calculate MAE with weights W

% Parameters of the function:
% --------------------------
% x  : the input of neural network
% y  : the label
% W  : the parameters of the neural network.
% 
% Returns:
% -------

% MAE: mean absolute error
% =========================================================================

function MAE = calMAE(W, x, y)
    inputW = W{1};
    outputW = W{2};
    H = feval(@logsig_m, [ones(num, 1) x] * inputW);%the output of hidden layer
    output = feval(@logsig_m, [ones(num,1) H] * outputW);%num * ydim, got through the second layer
    
    ymax = max(y);
    for i = 1 : num
        ind = find(output(i, :) < 0.5);
        if isempty(ind)
            yy(i) = ymax;
        else
            yy(i) = ind(1) - 1;
        end
    end
    
    MAE = sum(abs(y - yy)) / num;
    disp([' MAE:' num2str(MAE)]);
end