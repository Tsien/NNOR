%Author: Tsien at NUAA, China. 05/12/2015
%This function is to select the number of hidden neurons. 

% Parameters of the function:
% --------------------------
% K      : K fold cross validation, K = 1 means no CV.
% train_x: training samples
% train_y: the labels of training samples with CA code
% trainY : the original labels
% alph   : additional bias
% ymax   : max value of y
% Returns:
% -------
% hn   : the number of hidden neurons. 
%
% ===================================================================================================

function hn = getHidnum(k, train_x, train_y, trainY, alph, ymax)
    [num, x_dim] = size(train_x);
    [num, y_dim] = size(train_y);
    hidnum = zeros(3, 1);
    hidnum(1) = ceil(sqrt(x_dim + y_dim) + alph);%alph is in the range of [1, 10]
    hidnum(2) = ceil(log(x_dim) / log(2));
    hidnum(3) = ceil(sqrt(x_dim * y_dim));
    
    vali_num = ceil(num / k);
    train_num = num - vali_num;
    MAE = zeros(3, 1);
    for h = 1 : 3
        for i = 1 : k
            j = i * vali_num;
            vali_x = train_x(j - vali_num + 1 : min(j, end), :);
            vali_y = train_y(j - vali_num + 1 : min(j, end), :);
            valiY = trainY(j - vali_num + 1 : min(j, end));
            if i == 1
                cv_train_x = train_x(vali_num + 1 : end, :);
                cv_train_y = train_y(vali_num + 1 : end, :);
                cv_trainY = trainY(vali_num + 1 : end);
            else
                if i == k
                    cv_train_x = train_x(1 : j - vali_num, :);
                    cv_train_y = train_y(1 : j - vali_num, :);
                    cv_trainY = trainY(1 : j - vali_num);
                else
                    cv_train_x = [train_x(1 : j - vali_num, :); train_x(j + 1 : end, :)];
                    cv_train_y = [train_y(1 : j - vali_num, :); train_y(j + 1 : end, :)];
                    cv_trainY = [trainY(1 : j - vali_num); trainY(j + 1 : end)];
                end
            end
            cv_W{i} = elMseb(cv_train_x, cv_train_y, cv_trainY, hidnum(h));
            vali_MAE(i) = calMAE(cv_W{i}, vali_x, valiY, ymax);
        end
        MAE(h) = sum(vali_MAE) / k;
    end
    [mae, h] = min(MAE);
    hn = hidnum(h);
end