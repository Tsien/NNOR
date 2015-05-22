function newdata = CAcode(data)
%DATADEAL adapts target(# of rings) to fit ordinal regression
%e.g. turn data(i) = 5 to data(i) = [1 1 1 1 1 0 0 ...]
    m = numel(data);
    n = max(data);
    newdata = zeros(m, n);%rings = [1, 29]

    for i = 1 : m
        for j = 1 : data(i)%CA code [1 1 1.. 0 0...]
            newdata(i, j) = 1;
        end
    end
    
    %adapt to sigmoid function.
    newdata(find(newdata == 1)) = 0.95;
    newdata(find(newdata == 0)) = 0.05;
end

