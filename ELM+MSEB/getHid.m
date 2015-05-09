function hidnum = getHid(input, output, alph)
    hidnum = zeros(3, 1);
    hidnum(1) = ceil(sqrt(input + output) + alph);%alph is in the range of [1, 10]
    hidnum(2) = ceil(log(input) / log(2));
    hidnum(3) = ceil(sqrt(input * output));
    %hidnum = [10, 25, 50, 100];
end