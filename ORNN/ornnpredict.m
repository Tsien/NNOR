function yy = ornnpredict(nn, x, y, T)
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    
    [num, y_dim] = size(y);
    output = nn.a{end};
    ymax = max(y);
    yy = zeros(num, 1);
    for i = 1 : num
        ind = find(output(i, :) < T);
        if isempty(ind)
            yy(i) = ymax;
        else
            yy(i) = ind(1) - 1;
        end
    end
end