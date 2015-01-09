%Author: Tsien
%Parameter:
%d: the label of samples (size: 1*S)
%
%Output:
%y: the CA of label (size: max*S, max = max(d))
function y = CAcode(d)
  m = max(d);
  s = size(d, 2);
  y = zeros(m, s);
  for i=1:1:s
      n = d(i);
      for j=1:1:n
          y(j,i) = 0.75;
      end
      for j=n+1:1:m
          y(j,i) = 0.25;
      end
  end