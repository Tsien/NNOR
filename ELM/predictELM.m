%Author: Tsien at NUAA, China. 05/13/2015
%This function is to select the number of hidden neurons. 

% Parameters of the function:
% --------------------------
% x             - dataset
% y             - label
% W             - weights
%
% Returns:
% -------
% MAE           - Training and testing MAE
% MZOE          - Training and testing MZOE
%
% ===================================================================================================

function [MAE, MZOE] = predictELM(W, data)

    x = data(:, 1:end-1);
    T = data(:, end);
    InputWeight = W{1};
    OutputWeight = W{2};
    BiasofHiddenNeurons = W{3};
    
    [num, x_dim] = size(x);
    tempH = InputWeight * x';
    ind = ones(1, num);
    biasM = BiasofHiddenNeurons(:, ind);
    tempH = tempH + biasM;
    H = 1 ./ (1 + exp(-tempH));
    Y = H' * OutputWeight;
    
    MAE = sum(abs(T - Y)) / num;
    Y = ceil(Y - 0.5);
    MZOE = numel(find(T ~= Y)) / num;

    disp(['predicet--> MAE:' num2str(MAE) ', MZOE:' num2str(MZOE)]);


end