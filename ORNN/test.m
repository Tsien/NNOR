function [W, MAE, MZOE, names] = test(datafile)
    data = load(datafile);
    
    names = fieldnames(data);
    n = numel(names);
    MAE = zeros(n, 100, 2);
    MZOE = zeros(n, 100, 2);
    for tp = 1 : n
        tmpData = getfield(data, names{tp});
        for i = 1 : 100
            disp(['NO.' num2str(i)]);
            [W, MAE(tp, i, :), MZOE(tp, i, :)] = ORNN(tmpData);
        end
    end
end
