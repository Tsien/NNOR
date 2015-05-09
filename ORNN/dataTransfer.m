function data = dataTransfer(data)
%DATATRANSFET adapts data to fit ordinal regression
%e.g. x = [0 0 0 1 0 0 0] will be transfered to [1 1 1 1 0 0 0]

[m n] = size(data);%m = # of samples, n = # of features
for i = 1:m
    for j = 1:n
        if data(i, j)
            break;
        end
        data(i, j) = 1;
    end
end