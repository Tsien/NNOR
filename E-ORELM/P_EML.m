%Author: Tsien at NUAA, China. 05/08/2015
%This is the main function of the whole project
%The dataset contains one variable, called data. The last column of 
%data is the label. 

% Parameters of the function:
% --------------------------
% data: contains label in its last column
% K   : K fold cross validation, K = 1 means no CV.
% P   : try to ensemble P ELMs
% 
% Returns:
% -------
% W   : weights of every eELM.
% MAE : Mean Absolute Error of the whole system.
% Time: Training time and test time
% ===================================================================================================
function [W, MAE, MZOE, Time] = P_EML(data, K, P)
    x = data(:, 1:end - 1);%inputs of the network (size: m x d). m = #samples
    y = data(:, end);% original label
    ymax = max(y);
    cay = CAcode(y);% encoded label.
    [num, in_dim] = size(x);
    [num, out_dim] = size(cay);
    
    % =====================================================
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
    % =====================================================
    %select hidden number with K-fold cross validation
    hidnum = getHidnum(K, train_x, train_y, trainY, 3, ymax);
    % training P ELMs 
    W = cell(P, 1);
    stime = cputime;
    for p = 1 : P % train P ELMs
        W{p} = elMseb(train_x, train_y, trainY, hidnum);        
    end
    etime = cputime;
    Time(1) = etime - stime;
    
    % ======================================================
    % ensemble all P ELMs
    [num, x_dim] = size(train_x);
    [num, y_dim] = size(train_y);
    output = zeros(num, y_dim);
    for p = 1 : P
        inputW = W{p}{1};% x_dim * hidnum
        outputW = W{p}{2};% hidnum * y_dim
        H = feval(@logsig_m, [ones(num, 1) train_x] * inputW);
        output = output + feval(@logsig_m, [ones(num,1) H] * outputW);
    end
    
    output = output ./ P;    
    yy = zeros(num, 1);
    for i = 1 : num
        ind = find(output(i, :) < 0.5);
        if isempty(ind)
            yy(i) = ymax;
        else
            yy(i) = ind(1) - 1;
        end
    end
    
    MAE(1) = sum(abs(trainY - yy)) / num;
    MZOE(1) = numel(find(trainY ~= yy)) / num;
    disp(['Train MAE:' num2str(MAE(1)) ' Train MZOE:' num2str(MZOE(1))]);
    
    % ======================================================
    % ensemble all P ELMs
    [num, x_dim] = size(test_x);
    [num, y_dim] = size(test_y);
    output = zeros(num, y_dim);
    
    stime = cputime;
    for p = 1 : P
        inputW = W{p}{1};% x_dim * hidnum
        outputW = W{p}{2};% hidnum * y_dim
        H = feval(@logsig_m, [ones(num, 1) test_x] * inputW);
        output = output + feval(@logsig_m, [ones(num,1) H] * outputW);
    end
    etime = cputime;
    Time(2) = etime - stime;
    
    output = output ./ P;
    yy = zeros(num, 1);
    for i = 1 : num
        ind = find(output(i, :) < 0.5);
        if isempty(ind)
            yy(i) = ymax;
        else
            yy(i) = ind(1) - 1;
        end
    end
    MAE(2) = sum(abs(testY - yy)) / num;
    MZOE(2) = numel(find(testY ~= yy)) / num;
    disp(['Test MAE:' num2str(MAE(2)) ' Test MZOE:' num2str(MZOE(2))]);
end