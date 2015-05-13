%Author: Tsien at NUAA, China. 05/13/2015
%This function is to select the number of hidden neurons. 

% Parameters of the function:
% --------------------------
% K         : K fold cross validation, K = 1 means no CV.
% train_data: contains label in the last column
%
% Returns:
% -------
% hn   : the number of hidden neurons. 
%
% ===================================================================================================

function hn = getHidnum(k, train_data, alph)
    
    [num, x_dim] = size(train_data);
    hidnum = zeros(3, 1);
    hidnum(1) = ceil(sqrt(x_dim + 1) + alph);%alph is in the range of [1, 10]
    hidnum(2) = ceil(log(x_dim) / log(2));
    hidnum(3) = ceil(sqrt(x_dim));
    
    vali_num = ceil(num / k);
    MAE = zeros(3, 1);
    for h = 1 : 3
        for i = 1 : k
            j = i * vali_num;
            vali_data = train_data(j - vali_num + 1 : min(j, end), :);
            if i == 1
                cv_train_data = train_data(vali_num + 1 : end, :);
            else
                if i == k
                    cv_train_data = train_data(1 : j - vali_num, :);
                else
                    cv_train_data = [train_data(1 : j - vali_num, :); train_data(j + 1 : end, :)];
                end
            end
            [mae, mzoe, W] = myELM(cv_train_data, hidnum(h));
            [vali_MAE, mzoe] = predictELM(W, vali_data);
        end
        MAE(h) = sum(vali_MAE) / k;
    end
    [mae, h] = min(MAE);
    hn = hidnum(h);
end