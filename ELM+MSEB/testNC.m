function [good, er] = testNC(hidnum)
    load abalone-01-5;
    ORy = exCALabel(0.95, ORy);
    [num xdim] = size(x);
    index = randperm(num);
    num = ceil(num*0.25);
    
    train_x = x(index(1:num), :);
    train_y = ORy(index(1:num), :);
    test_x = x(index(num + 1:end), :);
    test_y = ORy(index(num + 1:end), :);
    testY = y(index(num + 1:end), :);
    %======================================================
    % normalize
       
    [train_x, mu, sigma] = zscore(train_x);
    test_x = normalize(test_x, mu, sigma);
    %======================================================
    %randomly choose input weights.
    inputW = randn(xdim + 1, hidnum);% subject to (0, 1) normal distribution
    H = feval(@logsig_m, [ones(num, 1) train_x] * inputW);%the first layer
    
    %======================================================
    %training
    ydim = size(train_y, 2);
    outputW = zeros(hidnum + 1, ydim);
    for j = 1 : ydim%traverse every output
        [A,b,outputW(:, j)] = nnopt2(H, train_y(:, j), @ilogsig, @dlogsig_m);
    end
    %======================================================
    %test
    [num, xdim] = size(test_x);
    H = feval( @logsig_m , [ones(num,1) test_x] * inputW ) ;%num * hidnum, go through the first layer
    out = feval(@logsig_m, [ones(num,1) H] * outputW);%num * ydim, got through the second layer
    
    ymax = max(testY);
    for i = 1 : num
        ind = find(out(i, :) < 0.5);
        if isempty(ind)
            ot(i, 1) = ymax;
        else
            ot(i, 1) = ind(1) - 1;
        end
    end
    
    dif = abs(testY - ot);
    good = find(dif < 3);
    er = zeros(2, 1);
    er(1) = numel(good) / num;
    er(2) = sum(dif) / num;
    disp(['CS: ' num2str(er(1)) ' MAE:' num2str(er(2))]);
    %======================================================

end