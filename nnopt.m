%Author: Tsien
%Parameter:
%   x: samples (size: I*S, I: #features, S: #samples)
%   d: the label of samples (size: T*S)
%
%Output:
%   W: the weights of NN (size: (I+1)*T)
function W = nnopt(x, d)
finv = @ilogsig;
fderiv = @dlogsig_m;
[I,S] = size(x);
out = size(d, 1);
W = zeros(I+1, out);
for i = 1 : 1 : out
    [A, b, W(:, i)] = nnopt2(x, d(i, :), finv, fderiv);
end
