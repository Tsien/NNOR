%Author: Tsien
%Parameter:
%   d: the label of samples (size: 1*S)
%   a: the left border of the interval
%   b: the right border of the interval
%
%Output:
%   y: the label of samples after being normalized in the interval [a, b](size: 1*S)

function y = normLabel(d, a, b)

y = a + (d - min(d))*(b - a)/(max(d) - min(d));