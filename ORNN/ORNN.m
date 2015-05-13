%Author: Tsien at NUAA, China. 05/10/2015
%This is the main function of ORNN
%The dataset contains one variable, called data. The last column of 
%data is the label. 

% Parameters of the function:
% --------------------------
% data  : contains label in its last column
% 
% Returns:
% -------
% W   : weights of neural network.
% MAE : Mean Absolute Error of the whole system.
%
% =========================================================================

function [W, MAE, MZOE] = ORNN(data)
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
    hidnum = getHidnum(10, train_x, train_y, trainY, 3);
    
    rand('state',0);
    nn = nnsetup([in_dim hidnum out_dim]);
    nn.activation_function = 'sigm';    %  Sigmoid activation function
    nn.learningRate = 1;                %  Sigm require a lower learning rate
    opts.numepochs =  10;               %  Number of full sweeps through data
    opts.batchsize = 100;               %  Take a mean gradient step over this many samples
    opts.plot      = 1;                 %  enable plotting
    nn = nntrain(nn, train_x, train_y, opts);
    [MAE, MZOE] = ornntest(nn, test_x, testY);
    W = nn.W;
    disp(['MAE: ' num2str(MAE) ', MZOE: ' num2str(MZOE)]);
end