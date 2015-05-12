%Author: Tsien at NUAA, China. 05/10/2015
%This is the main function of the whole project
%The dataset contains one variable, called data. The last column of 
%data is the label. 

% Parameters of the function:
% --------------------------
% data: contains label in its last column
% 
% Returns:
% -------
% W   : weights of neural network.
% MAE : Mean Absolute Error of the whole system.
%
% ===================================================================================================
function [W, MAE] = MSEB(data)
    x = data(:, 1:end - 1);%inputs of the network (size: m x d). m = #samples
    y = data(:, end);% original label
    cay = CAcode(y);% encoded label.
    clear data;
    
    [num, in_dim] = size(x);
    [num, out_dim] = size(cay);
    
    index = randperm(num);
    test_num = ceil(0.2 * num);
    test_x = x(index(1 : test_num), :);
    test_y = cay(index(1 : test_num), :);
    testY = y(index(1 : test_num));
    train_x = x(index(test_num + 1 : end), :);
    train_y = cay(index(test_num + 1 : end), :);
    trainY = y(index(test_num + 1 : end));
    %======================================================
    % normalize   
    [train_x, mu, sigma] = zscore(train_x);
    test_x = normalize(test_x, mu, sigma);
    
    %======================================================
    yd = size(train_y, 2);
    for j = 1 : yd
        [A,b,W(:, j)] = nnopt2(train_x, train_y(:, j), @ilogsig, @dlogsig_m);
    end

    [test_num, dim] = size(test_x);
    output = feval( @logsig_m , [ones(test_num, 1) test_x]*W ) ;

    %======================================================
    %Ordinal Regression
    ymax = max(testY);
    yy = zeros(test_num, 1);
    for i = 1 : test_num
        ind = find(output(i, :) < 0.5);
        if isempty(ind)
            yy(i) = ymax;
        else
            yy(i) = ind(1) - 1;
        end
    end
   
    MAE = sum(abs(yy - testY)) / test_num;    
    disp([' MAE:' num2str(MAE)]);
    %======================================================

end

