function newdata = exCALabel(one, dataset)
    % transform the [1,1,1,...,0,0,0] CAcode to [one,one,one,...,1-one]
    % formals
    newdata = zeros(size(dataset));
    newdata(find(dataset == 1)) = one;
    newdata(find(dataset == 0)) = 1 - one;
    
end
