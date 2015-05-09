function ORNN
%load mnist_uint8;
load abalone2;
% train_x = double(train_x) / 255;
% test_x  = double(test_x)  / 255;
% train_y = double(train_y);
% test_y  = double(test_y);
% train_y = dataTransfer(train_y);
% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

% vx   = train_x(1:10000,:);
% tx = train_x(10001:end,:);
% vy   = train_y(1:10000,:);
% ty = train_y(10001:end,:);

% vx = train_x(2401:end, :);
% tx = train_x(1:2400, :);
% vy = train_ORy(2401:end, :);
% ty = train_ORy(1:2400, :);

rand('state',0)
nn = nnsetup([30 100 1]);
nn.activation_function = 'sigm';    %  Sigmoid activation function
nn.learningRate = 1;                %  Sigm require a lower learning rate
opts.numepochs =  10;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
opts.plot      = 1;    %  enable plotting
%nn = nntrain(nn, tx, ty, opts, vx, vy);   %  nntrain takes validation set as last two arguments (optionally)
nn = nntrain(nn, train_x, train_ORy, opts);
[er, bad] = ornntest(nn, test_x, test_y);
disp(['vanilla neural net test error: ' num2str(er)]);
assert(er < 0.08, 'Too big error!');
