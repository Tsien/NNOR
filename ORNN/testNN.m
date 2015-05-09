function testNN
load FgNetPoints;
%load mnist_uint8;
%load abalone;
% train_x = double(train_x) / 255;
% test_x  = double(test_x)  / 255;
% train_y = double(train_y);
% test_y  = double(test_y);
% normalize
train_x = X;
train_y = Y;
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);
% split training data into training and validation data
vx   = train_x(1:10000,:);
tx = train_x(10001:end,:);
vy   = train_y(1:10000,:);
ty = train_y(10001:end,:);

% vx = train_x(2401:end, :);
% tx = train_x(1:2400, :);
% vy = train_NNy(2401:end, :);
% ty = train_NNy(1:2400, :);

rand('state',0)
nn                      = nnsetup([784 100 10]);
nn.activation_function  = 'sigm';    %  Sigmoid activation function
%nn.output               = 'softmax';                   %  use softmax output
nn.learningRate         = 1;                %  Sigm require a lower learning rate
opts.numepochs          = 10;                           %  Number of full sweeps through data
opts.batchsize          = 100;                        %  Take a mean gradient step over this many samples
opts.plot               = 1;                           %  enable plotting
nn = nntrain(nn, tx, ty, opts, vx, vy);                %  nntrain takes validation set as last two arguments (optionally)

[er, bad] = nntest(nn, test_x, test_y);
disp(['NN with sigmoid activation - test error: ' num2str(er)]);
assert(er < 0.1, 'Too big error');
