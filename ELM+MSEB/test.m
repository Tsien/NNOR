function [Time, W, MAE, MZOE, names] = test(iteration, datafile, K, P)
    data = load(datafile);
    
    names = fieldnames(data);
    n = numel(names);
    names = sort(names);
    MAE = zeros(n, 4);
    MZOE = zeros(n, 4);
    Time = zeros(n, 4);
    mae = zeros(iteration, 2);
    mzoe = zeros(iteration, 2);
    time = zeros(iteration, 2);
    for tp = 1 : n
        tmpData = getfield(data, names{tp});
        for i = 1 : iteration
            disp([names(tp) '--NO.' num2str(i)]);
            [W, mae(i, :), mzoe(i, :), time(i, :)] = P_EML(tmpData, K, P);
        end
        MAE(tp, 1) = mean(mae(:, 1));
        MAE(tp, 2) = std(mae(:, 1));
        MAE(tp, 3) = mean(mae(:, 2));
        MAE(tp, 4) = std(mae(:, 2));
        MZOE(tp, 1) = mean(mzoe(:, 1));
        MZOE(tp, 2) = std(mzoe(:, 1));
        MZOE(tp, 3) = mean(mzoe(:, 2));
        MZOE(tp, 4) = std(mzoe(:, 2));
        Time(tp, 1) = mean(time(:, 1));
        Time(tp, 2) = std(time(:, 1));
        Time(tp, 3) = mean(time(:, 2));
        Time(tp, 4) = std(time(:, 2));
    end
    save result Time W MAE MZOE names;
end
