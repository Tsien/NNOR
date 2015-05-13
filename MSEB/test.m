function [W, MAE, MZOE, names] = test(datafile)
    data = load(datafile);
    
    names = fieldnames(data);
    n = numel(names);
    MAE = zeros(n, 2);
    MZOE = zeros(n, 2);
    mae = zeros(100, 2);
    mzoe = zeros(100, 2);
    for tp = 1 : n
        tmpData = getfield(data, names{tp});
        for i = 1 : 100
            disp(['NO.' num2str(i)]);
            [W, mae(i, :), mzoe(i, :)] = MSEB(tmpData);
        end
        MAE(tp, :) = mean(mae);
        MZOE(tp, :) = mean(mzoe);
    end
    save result W MAE MZOE names;
end
