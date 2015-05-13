%Author: Tsien at NUAA, China. 05/13/2015
%This is the main function to caculate MAE, MZOE
% --------------------------
% x   : the input of neural network
% y   : the label
% W   : the parameters of the neural network.
% ymax: the number of classes
% 
% Returns:
% -------

% MAE : mean absolute error
% MZOE: mean zero one error
%
% ===================================================================================================

function [MAE, MZOE] = predict(W, x, y, ymax)

    [num, x_dim] = size(x);
    output = feval(@logsig_m, [ones(num,1) x] * W);%num * ydim
    
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
    disp([' MAE:' num2str(MAE) ', MZOE:' num2str(MZOE)]);

end



