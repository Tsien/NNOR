%Author: Tsien at NUAA, China. 05/13/2015
%This function is to select the number of hidden neurons. 

% Parameters of the function:
% --------------------------
% data                  - dataset
%
% Returns:
% -------
% MAE           - Training and testing MAE
% MZOE          - Training and testing MZOE
%
% ===================================================================================================


function [MAE, MZOE] = mainELM(data)

    [num, dim] = size(data);
    index = randperm(num);
    test_num = ceil(0.2 * num);
    train_data = data(index(test_num + 1 : end), :);
    test_data = data(index(1 : test_num), :);
    
    hidnum = getHidnum(10, train_data, 3);
    [MAE(1), MZOE(1), W] = myELM(train_data, hidnum);
    [MAE(2), MZOE(2)] = predictELM(W, test_data);
end

