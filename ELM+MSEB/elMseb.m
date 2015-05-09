% Author: Tsien at NUAA, China. 05/08/2015
% Description:
% This function is the main function of my model
% It is a Single Layer Feedforward neural network(SLFN). The input weights 
% connecting input layer and hidden layer are random chosen. The output
% weights connecting hidden layer and output layer are analytically determined
% by using MSEB, which is a convex function.

% Parameters of the function:
% --------------------------
% x     : the input of neural network
% cay   : the encoded label
% y     : the label
% hidnum: the number of neurons in hidden layer
% 
% Returns:
% -------
% W  : the parameters of the neural network.
% MAE: mean absolute error
% =========================================================================

function [W, MAE] = elMseb(x, cay, y, hidnum)
    [num, x_dim] = size(x);
    [num, y_dim] = size(cay);
    % ======================================================
    %randomly choose input weights.
    inputW = randn(x_dim + 1, hidnum);% subject to (0, 1) normal distribution
    H = feval(@logsig_m, [ones(num, 1) x] * inputW);%the output of hidden layer
    
    % ======================================================
    % training
    outputW = zeros(hidnum + 1, y_dim);
    for j = 1 : y_dim%traverse every output
        [A,b,outputW(:, j)] = nnopt2(H, cay(:, j), @ilogsig, @dlogsig_m);
    end
    
    W{1} = inputW;
    W{2} = outputW;
    
    %======================================================
    %calculate MAE
    MAE = calMAE(W, x, y);
end