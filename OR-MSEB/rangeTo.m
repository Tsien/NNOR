%Author: Tsien at NUAA, China. 05/10/2015
%This function transform data to range of [a, b]
%The data variable is a column vector. 

% Parameters of the function:
% --------------------------
% data: data that needed to be normalized, a column vector
% a   : left limits
% b   : right limits
% Returns:
% -------
% newdata   : normalied data
%
% ===================================================================================================

function newdata = rangeTo(data, a, b)
    mind = min(data);
    maxd = max(data);
    newdata = (data - mind) * (b - a) / (maxd - mind) + a;
end