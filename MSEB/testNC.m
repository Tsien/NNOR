function [bad er] = testNC(n)
    load abalone-ok;
     [m d] = size(x);
     index = randperm(m);
     m = ceil(m*0.8);
    
    train_x = x(index(1:m), :);
    train_y = ORy(index(1:m), :);
    test_x = x(index(m + 1:end), :);
    test_y = ORy(index(m + 1:end), :);
    testY = y(index(m + 1:end), :);
    % normalize
       
    [train_x, mu, sigma] = zscore(train_x);
    test_x = normalize(test_x, mu, sigma);
    
    res = train_x;
    yd = size(train_y, 2);
    for i = 1 : n
        for j = 1 : yd
            [A,b,W{i}(:, j)] = nnopt2(res, train_y(:, j), @ilogsig, @dlogsig_m);
        end
        [m, d] = size(res);
        res = feval( @logsig_m , [ones(m,1) res]*W{i} ) ;
    end

    [m, d] = size(test_x);
    out = test_x;
    for i = 1 : n
        out = feval( @logsig_m , [ones(m,1) out]*W{i} ) ;
    end
    
%     ot(find(out > 0.5)) = 0.95;%threshold = 0.5
%     ot(find(out <= 0.5)) = 0.05;

    %======================================================
    %Ordinal Regression
    [m, d] = size(out);
    ot = zeros(m, 1);
    for i = 1 : m
        pos = find(out(i, :) < 0.5);
        if isempty(pos)
            ot(i) = d;
        else
            ot(i) = pos(1) - 1;
        end
    end
    out = ot;
    ot = abs(ot - testY);
    bad = find(ot >= 3);
    er = zeros(2, 1);
    er(1) = 1 - numel(bad) / m;
    er(2) = sum(ot) / m;
    disp(['CS: ' num2str(er(1)) ' MAE:' num2str(er(2))]);
    %======================================================

    %bad = find(ot' ~= test_y);    
%     er = zeros(2, 1);
%     er(1) = numel(bad) / m;
%     er(2) = sum((out - test_y).^2) / m;
% 
%     disp(['test error: ' num2str(er(1)) ' MSE:' num2str(er(2))]);
end

