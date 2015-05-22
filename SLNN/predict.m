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

    num = size(x, 1);
    output = feval(@logsig_m, [ones(num,1) x] * W);%num * ydim
    output = rangeTo(output, 1, ymax);
    y = rangeTo(y, 1, ymax);
    yy = ceil(output - 0.5);
    MAE = sum(abs(y - yy)) / num;
    MZOE = numel(find(y ~= yy)) / num;
    disp([' MAE:' num2str(MAE) ', MZOE:' num2str(MZOE)]);

end



