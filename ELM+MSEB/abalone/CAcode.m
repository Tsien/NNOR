function newdata = CAcode(data, opt)
%DATADEAL adapts target(# of rings) to fit ordinal regression
%e.g. turn data(i) = 5 to data(i) = [1 1 1 1 1 0 0 ...]
    m = numel(data);
    n = max(data);
    newdata = zeros(m, n);%rings = [1, 29]

    if opt == 1
        for i = 1 : m
            for j = 1 : data(i)%CA code [1 1 1.. 0 0...]
                newdata(i, j) = 1;
            end
        end
    else
        for i = 1 : m%normal [0 0 1 0 0 ...]
            newdata(i, data(i)) = 1;
        end
    end
end

