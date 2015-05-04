function [bad er] = testMC(n)
    load wdbc;
    [m d] = size(x);
    index = randperm(m);
    m = ceil(m*0.6);
    
    train_x = x(index(1:m), :);
    train_y = y(index(1:m), :);
    test_x = x(index(m:end), :);
    test_y = y(index(m:end), :);
    % normalize
    [train_x, mu, sigma] = zscore(train_x);
    test_x = normalize(test_x, mu, sigma);
    
    res = train_y;
    for i = 1 : n
        [m d] = size(res);
        for j = 1 : d
            [A,b,W{i}(:, j)] = nnopt2(train_x, res(:, d), @ilogsig, @dlogsig_m);
        end

        w = W{i}(2:end, :);
        res  = res * w';
    end

    [m, d] = size(test_x);
    out = test_x;
    for i = n :-1: 1
        out = feval( @logsig_m , [ones(m,1) out]*W{i} ) ;
    end
    
    ot(find(out > 0.5)) = 0.95;%threshold = 0.5
    ot(find(out <= 0.5)) = 0.05;

    bad = find(ot' ~= test_y);    
    er = zeros(2, 1);
    er(1) = numel(bad) / m;
    er(2) = sum((out - test_y).^2) / m;

    disp(['test error: ' num2str(er(1)) ' MSE:' num2str(er(2))]);
end

