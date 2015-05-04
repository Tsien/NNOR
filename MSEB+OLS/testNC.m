function [bad er] = testNC
    load abalone-ok;
     [m d] = size(x);
     index = randperm(m);
     m = ceil(m*0.8);
    
    train_x = x(index(1:m), :);
    train_y = ORy(index(1:m), :);
    test_x = x(index(m + 1:end), :);
    test_y = ORy(index(m + 1:end), :);
    testY = y(index(m + 1:end), :);
    %======================================================
    % normalize
       
    [train_x, mu, sigma] = zscore(train_x);
    test_x = normalize(test_x, mu, sigma);
    %======================================================
    
    %======================================================
    %training
    [n, m] = size(train_x);
    yd = size(train_y, 2);
    W = zeros(m+1, yd);
    for j = 1 : yd
        [A,b,W(:, j)] = nnopt2(train_x, train_y(:, j), @ilogsig, @dlogsig_m);
    end
    %======================================================
    %solve ICLS problem
    H = feval(@logsig_m, [ones(n, 1) train_x]*W);%n*yd
    C = zeros(yd, yd);%yd*yd
    for i = 1:yd
        for j = 1:i
            C(i, j) = 1;
        end
    end
    
    hatY = train_y * inv(C');%n*yd
    delta = pinv([ones(n, 1) H]) * hatY;%(yd + 1)*yd
    delta(find(delta < 0)) = 0;
    beta = C * delta';%yd*(yd+1)
    %======================================================
    %test
    [m, d] = size(test_x);
    H = feval( @logsig_m , [ones(m,1) test_x]*W ) ;%m*d
    out = [ones(m,1) H] * beta';%n*yd;
    for i = 1 : yd
        iprob(:, i) = out(:, i) ./ out(:, end);
    end
    for i = 2 : yd
        prob(:, i) = iprob(:, i) - iprob(:, i - 1);
    end
    for i = 1 : m
        [val predY(i)] = max(prob(i, :));
    end
    
    dif = abs(testY - predY');
    bad = find(dif >= 3);

    disp(['CS: ' num2str(er(1)) ' MAE:' num2str(er(2))]);
    %======================================================

end

