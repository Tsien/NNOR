function dataPre

%    load abalone2.new;%F&M = [1 0], I = [0 1]
    load tmp.mat;
    dim = 3;
    height = abalone(:, 3 + dim);%height is 4th feature, the 1st is sex
    %elimate height == 1.13
    abalone(find(height == 1.13), 3 + dim) = 0.13;%2052

    %elimate height == 0
    abalone(find(height == 0), :) = [];
    
    
    preX = abalone(:, 1 + dim:end - 1);
    whole = preX(:, 4);
    shucked = preX(:, 5);
    viscera = preX(:, 6);
    shell = preX(:, 7);
    %Whole Weight = Shucked Weight + Viscera Weight + Shell Weight + Unknown mass of water/blood lost from shucking process
    m = find(whole <= shucked + viscera + shell);
    abalone(m, :) = [];
    
    index = randperm(size(abalone, 1));
    x = abalone(index(1:4000), 1:end - 1);
    y = abalone(index(1:4000), end);
    NNy = CAcode(y, 0);
    ORy = CAcode(y, 1);

    save abalone-01 x y NNy ORy;
end

