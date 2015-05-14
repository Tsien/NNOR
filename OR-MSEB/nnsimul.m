% ======================================================================================
% nnsimul
% ------------------------------
% Written by Oscar Fontenla-Romero, 22.03.2006 @ LIDIA,UDC - A Coruña, Spain
% This code is not optimized for speed etc. in any sense, use at your own risk
%
% PURPOSE OF THIS CODE:
% It obtains the outputs of a one-layer artificial neural network.
% ======================================================================================
%  
% Syntax:
% ------
% Output = nnsimul(W,x,f)
%
% Parameters of the function:
% --------------------------
% W : weights of the neural network (size: I+1 x 1). The 1st element is the bias (wj0).
% x : inputs of the network (size: I x S).
% f : neural function. 
% 
% Returns:
% -------
% Output : outputs of the network for all the input data.
%
% ======================================================================================
function Output = nnsimul(W,x,f)

% Number of inputs (I) and data points (S)
[I,S]=size(x);

% Neural Network Simulation
Output = feval( f , W'*[ones(1,S); x] ) ;
