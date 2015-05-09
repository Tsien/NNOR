%Author: Tsien at NUAA, China. 05/08/2015
%This is the main function of the whole project
%The dataset contains one variable, called data. The last column of 
%data is the label. 

% Parameters of the function:
% --------------------------
% dataset: contains label in its last column
% K      : K fold cross validation, K = 1 means no CV.
% P      : try to ensemble P ELMs
% 
% Returns:
% -------
% A : Coefficient matrix of the system A*w = b (size: d+1 x d+1).
% b : Right hand side values of the system A*w = b (size: d+1 x 1).
% w : Weight vector (size: d+1 x 1).
%     The 1st element is the bias (wj0).
%
% ===================================================================================================
function [good, er] = testNC(dataset, K, P)
    load(dataset);
    x = data(:, 1:end - 1);%inputs of the network (size: m x d). m = #samples
    y = data(:, end);% original label
    cay = CAcode(y);% encoded label.
    [num, in_dim] = size(x);
    [num, out_dim] = size(cay);
    hidnum = getHid(in_dim, out_dim, 5);
    
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
    % training P ELMs with K-fold cross validation
    nhid = numel(hidnum);
    train_MAE = zeros(P, 1);
    for p = 1 : P % train P ELMs
        best_MAE = zeros(nhid, 1);
        best_W = zeros(nhid, 1);
        for h = 1 : nhid % NN with different # nerons in hidden layer
            if K > 1
                vali_num = (num - test_num) / K;
                train_num = num - test_num - train_num;
                cv_MAE = zeros(K, 1)
                for i = 1 : K
                    j = i * vali_num;
                    vali_x = train_x(j - valinum + 1 : j, :);
                    vali_y = train_y(j - valinum + 1 : j, :);
                    valiY = trainY(j - valinum + 1 : j);
                    if i == 1
                        train_x = train_x(vali_num + 1 : end, :);
                        train_y = train_y(vali_num + 1 : end, :);
                        trainY = trainY(vali_num + 1 : end);
                    else
                        train_x = [train_x(1 : j - valinum, :) train_x(j + 1 : end, :)];
                        train_y = [train_y(1 : j - valinum, :) train_y(j + 1 : end, :)];
                        trainY = [trainY(1 : j - valinum) trainY(j + 1 : end)];
                    end
                    [cv_W{i}, cv_MAE(i)] = elMseb(train_x, train_y, trainY, hidnum(h));
                    vali_MAE(i) = calMAE(cv_W{i}, vali_x, vali_y);
                end
                [best_MAE(h), pos] = min(vali_MAE);
                best_W{h} = cv_W{pos};
            end            
        end
        [train_MAE(p), pos] = min(best_MAE);
        W{p} = best_W{pos};
    end
    
    % ======================================================
    % ensemble all P ELMs
    [num, x_dim] = size(test_x);
    output = zeros(num, P);
    for p = 1 : P
        inputW = W{p}{1};
        outputW = W{p}{2};
        H = feval(@logsig_m, [ones(num, 1) x] * inputW);
        output(:, p) = feval(@logsig_m, [ones(num,1) H] * outputW);
    end
    
    output_y = mean(output, 2);
    ymax = max(testY);
    yy = zeros(num, 1);
    for i = 1 : num
        ind = find(output_y(i, :) < 0.5);
        if isempty(ind)
            yy(i) = ymax;
        else
            yy(i) = ind(1) - 1;
        end
    end
    
    MAE = sum(abs(testY - yy)) / num;
    disp([' MAE:' num2str(MAE)]);
end