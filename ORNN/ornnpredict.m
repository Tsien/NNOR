function labels = ornnpredict(nn, x, T)
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    
    res = nn.a{end};
    [m n] = size(res);
    labels = zeros(m, 1);
    for i = 1:m
        labels(i) = 10;
        for j = 1:n
            if res(i, j) <= T
                if j > 1
                    labels(i) = j - 1;
                else
                    labels(i) = 1;
                end
                break;
            end
        end
    end
end