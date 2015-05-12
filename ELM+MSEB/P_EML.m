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
% W   : weights of every eELM.
% MAE : Mean Absolute Error of the whole system.
%
% ===================================================================================================
function [W, MAE] = P_EML(data, K, P)
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
        for h = 1 : nhid % NN with different # nerons in hidden layer
            if K > 1
                vali_num = ceil((num - test_num) / K);
                train_num = num - test_num - vali_num;
                cv_MAE = zeros(K, 1);
                for i = 1 : K
                    j = i * vali_num;
                    vali_x = train_x(j - vali_num + 1 : min(j, end), :);
                    vali_y = train_y(j - vali_num + 1 : min(j, end), :);
                    valiY = trainY(j - vali_num + 1 : min(j, end));
                    if i == 1
                        cv_train_x = train_x(vali_num + 1 : end, :);
                        cv_train_y = train_y(vali_num + 1 : end, :);
                        cv_trainY = trainY(vali_num + 1 : end);
                    else
                        if i == K
                            cv_train_x = train_x(1 : j - vali_num, :);
                            cv_train_y = train_y(1 : j - vali_num, :);
                            cv_trainY = trainY(1 : j - vali_num);
                        else
                            cv_train_x = [train_x(1 : j - vali_num, :); train_x(j + 1 : end, :)];
                            cv_train_y = [train_y(1 : j - vali_num, :); train_y(j + 1 : end, :)];
                            cv_trainY = [trainY(1 : j - vali_num); trainY(j + 1 : end)];
                        end
                    end
                    [cv_W{i}, cv_MAE(i)] = elMseb(cv_train_x, cv_train_y, cv_trainY, hidnum(h));
                    vali_MAE(i) = calMAE(cv_W{i}, vali_x, valiY);
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
    [num, y_dim] = size(test_y);
    output = zeros(num, y_dim);
    for p = 1 : P
        inputW = W{p}{1};% x_dim * hidnum
        outputW = W{p}{2};% hidnum * y_dim
        H = feval(@logsig_m, [ones(num, 1) test_x] * inputW);
        output = output + feval(@logsig_m, [ones(num,1) H] * outputW);
    end
    
    output = output ./ P;
    ymax = max(testY);
    yy = zeros(num, 1);
    for i = 1 : num
        ind = find(output(i, :) < 0.5);
        if isempty(ind)
            yy(i) = ymax;
        else
            yy(i) = ind(1) - 1;
        end
    end
    
    MAE = sum(abs(testY - yy)) / num;
    disp(['The test MAE:' num2str(MAE)]);
end