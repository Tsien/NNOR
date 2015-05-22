% ======================================================================================
% nnsimul
% ------------------------------
% Modified by Tsien, 16.04.2015 @ NUAA - Nanjing, China
% This code is not optimized for speed etc. in any sense, use at your own risk
%
% PURPOSE OF THIS CODE:
% It obtains the outputs of a one-layer artificial neural network.
% ======================================================================================
function [Output er]= nnsimul(W,x,y,f)
    %size(W) = (d+1)*1
    % Number of samples(m) and features(d)
    [m,d]=size(x);

    % Neural Network Simulation
    Output = feval( f , [ones(m, 1) x]*W ) ;
    ot = Output;
    ot(find(Output > 0.5)) = 0.95;%threshold = 0.5
    ot(find(Output <= 0.5)) = 0.05;
    
    bad = find(ot ~= y);    
    er = zeros(2, 1);
    er(1) = numel(bad) / size(x, 2);
    er(2) = sum((Output - y).^2) / numel(y);

end

