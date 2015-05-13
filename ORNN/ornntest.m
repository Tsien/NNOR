function [MAE, MZOE] = ornntest(nn, x, y)
    labels = ornnpredict(nn, x, y, 0.5);%T = 0.5;
    MAE = sum(abs(labels - y)) / numel(y);
    MZOE = numel(find(labels ~= y)) / numel(y);
end
