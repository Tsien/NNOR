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

function hn = getHidnum(k, train_x, train_y, trainY, alph)
    
    [num, x_dim] = size(train_x);
    [num, y_dim] = size(train_y);
    hidnum = zeros(3, 1);
    hidnum(1) = ceil(sqrt(x_dim + y_dim) + alph);%alph is in the range of [1, 10]
    hidnum(2) = ceil(log(x_dim) / log(2));
    hidnum(3) = ceil(sqrt(x_dim * y_dim));
    
    vali_num = ceil(num / k);
    MAE = zeros(3, 1);
    vali_MAE = zeros(k, 1);
    for h = 1 : 3
        for i = 1 : k
            j = i * vali_num;
            vali_x = train_x(j - vali_num + 1 : min(j, end), :);
            valiY = trainY(j - vali_num + 1 : min(j, end), :);
            if i == 1
                cv_train_x = train_x(vali_num + 1 : end, :);
                cv_train_y = train_y(vali_num + 1 : end, :);
                cv_trainY = trainY(vali_num + 1 : end, :);
            else
                if i == k
                    cv_train_x = train_x(1 : j - vali_num, :);
                    cv_train_y = train_y(1 : j - vali_num, :);
                    cv_trainY = trainY(1 : j - vali_num, :);
                else
                    cv_train_x = [train_x(1 : j - vali_num, :); train_x(j + 1 : end, :)];
                    cv_train_y = [train_y(1 : j - vali_num, :); train_y(j + 1 : end, :)];
                    cv_trainY = [trainY(1 : j - vali_num, :); trainY(j + 1 : end, :)];
                end
            end
            nn = nnsetup([x_dim hidnum(h) y_dim]);
            nn.activation_function = 'sigm';    %  Sigmoid activation function
            nn.learningRate = 1;                %  Sigm require a lower learning rate
            opts.numepochs =  10;               %  Number of full sweeps through data
            opts.batchsize = 100;               %  Take a mean gradient step over this many samples
            opts.plot      = 0;                 %  enable plotting
            nn = nntrain(nn, cv_train_x, cv_train_y, opts);
            vali_MAE(i) = ornntest(nn, vali_x, valiY);
        end
        MAE(h) = sum(vali_MAE) / k;
    end
    [mae, h] = min(MAE);
    hn = hidnum(h);
end