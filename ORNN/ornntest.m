function [er, bad] = ornntest(nn, x, y)
    labels = ornnpredict(nn, x, 0.5);%T = 0.5;
    %[dummy, expected] = max(y,[],2);
    expected = abs(y - labels);% Cumulative Score, l = 3
    bad = find(expected >= 3);    
    er = numel(bad) / size(x, 1);
    bad = sum(expected) / size(x, 1);
end
