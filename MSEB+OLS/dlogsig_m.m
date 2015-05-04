function result = dlogsig_m(x)

result = 1./(1+exp(-x)).^2.*exp(-x);

% Antes esta ... Esta mal!! 
result1 = (-exp(-x)) ./ (1+exp(-x)).^2;
del = result - result1;
