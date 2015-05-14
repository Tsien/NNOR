%Author: Tsien at NUAA, China. 05/14/2015
%This is the main function of the whole project
%The dataset contains one variable, called data. The last column of 
%data is the label. 

% Parameters of the function:
% --------------------------
% data: contains label in its last column
% 
% Returns:
% -------
% Time: training time and test time
% W   : weights of neural network.
% MAE : Mean Absolute Error of the whole system.
% MZOE: Mean Zero one Error
%
% ===================================================================================================
function [Time, W, MAE, MZOE] = MSEB(data)
    x = data(:, 1:end - 1);%inputs of the network (size: m x d). m = #samples
    y = data(:, end);% original label
    ymax = max(y);
    y = rangeTo(y, 0.05, 0.95)
    clear data;
    
    num = size(x, 1);
    
    index = randperm(num);
    test_num = ceil(0.2 * num);
    test_x = x(index(1 : test_num), :);
    test_y = y(index(1 : test_num), :);
    train_x = x(index(test_num + 1 : end), :);
    train_y = y(index(test_num + 1 : end), :);
    %======================================================
    % normalize   
    [train_x, mu, sigma] = zscore(train_x);
    test_x = normalize(test_x, mu, sigma);
    
    %======================================================
    %training
    stime = cputime;
    [A,b,W] = nnopt2(train_x, train_y, @ilogsig, @dlogsig_m);
    etime = cputime;
    Time(1) = etime - stime;
    [MAE(1), MZOE(1)] = predict(W, train_x, train_y, ymax);
    
    %======================================================
    %Ordinal Regression
    stime = cputime;
    [MAE(2), MZOE(2)] = predict(W, test_x, test_y, ymax);
    etime = cputime;
    Time(2) = etime - stime;
    %======================================================

end

