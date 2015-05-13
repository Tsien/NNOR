% Author: Tsien at NUAA, China. 05/09/2015
% Description:
% This function is to calculate MAE with weights W

% Parameters of the function:
% --------------------------
% x   : the input of neural network
% y   : the label
% W   : the parameters of the neural network.
% ymax: the number of classes
% 
% Returns:
% -------

% MAE: mean absolute error
% =========================================================================

function [MAE, MZOE] = calMAE(W, x, y, ymax)
    [num, x_dim] = size(x);
    inputW = W{1};
    outputW = W{2};
    H = feval(@logsig_m, [ones(num, 1) x] * inputW);%the output of hidden layer
    output = feval(@logsig_m, [ones(num,1) H] * outputW);%num * ydim, got through the second layer
    
    for i = 1 : num
        ind = find(output(i, :) < 0.5);
        if isempty(ind)
            yy(i, 1) = ymax;
        else
            yy(i, 1) = ind(1) - 1;
        end
    end
    
    MAE = sum(abs(y - yy)) / num;
    MZOE = numel(find(y ~= yy)) / num;
    disp([' MAE: ' num2str(MAE) ', MZOE: ' num2str(MZOE)]);
end