% ===================================================================================================
% nnopt algorithm - Version 4.0
% ------------------------------
% Modified by Tsien, 04.16.2015 @ NUAA - Nanjing, China
% Optimized for speed.
%
% PURPOSE OF THIS CODE:
% It trains an one-layer artificial neural network using a system of linear equations.
% The cost function assumed is MSE ( J = E[(d-y)'(d-y)] ).
% This version of the code allows incremental learning.
% ===================================================================================================
%  
% Syntax:
% ------
% [A,b,w] = nnopt2(x,d,finv,fderiv)
%
% Parameters of the function:
% --------------------------
% x      : inputs of the network (size: m x d). m = #samples
% d      : desired outputs of the network (size: m x 1).
% finv   : inverse of the neural function.
% fderiv : derivative of the neural function
% 
% Returns:
% -------
% A : Coefficient matrix of the system A*w = b (size: d+1 x d+1).
% b : Right hand side values of the system A*w = b (size: d+1 x 1).
% w : Weight vector (size: d+1 x 1).
%     The 1st element is the bias (wj0).
%
% ===================================================================================================

function [A,b,w] = nnopt2(x,y,finv,fderiv)

% Checking the number of arguments
if (nargin == 6 || nargin < 4)
    if (nargin == 6)
        disp('Error: You must supply both the A matrix and b vector.');
    else
        disp('Error: The function must have at least four arguments.');
    end
    % The outputs of the function are empty
    A = [];
    b = [];
    w = [];
else
    % Number of samples (m) and features (d)
    [m,d] = size(x);%m*d    
    % The bias is included as the first input (first row)
    x = [ones(m,1) x];%m*(d+1)    
    % Inverse of the neural function
    f_d = feval(finv,y); %y:desired outputs of the network (size: m x 1).    
   
    % System of linear equations: A*w = b    
    df2 = feval(fderiv,f_d).^2; % Square of the derivate of the neural function, m*1            
    b = x'*(f_d.*df2);          % b vector for the system of equetions, (d+1)*1
    df2 = df2*repmat(1,1,d+1);  %m*(d+1)
    A = x'*(df2.*x);            % A matrix for the system of equations, (d+1)*(d+1)                       

    % If the option flag is not supplied then is determined using the
    % condition number and the reciprocal condition estimator of the matrix A
        % If the matrix A is near to singular
        if ( (cond(A)<1e+016) && (rcond(A)>1e-015) )        
            option = 0;  % Regular solution
        else
            option = 1;  % Solution based on the Moore-Penrose pseudoinverse
        end
    
    % Solution to the equation A*w = b:    
    % - If A is square and not singular then the best solution is to use the inverse of A.
    % - If A is square and singular, then inv(A) does not exist. In this case, pinv(A) has some of, but not all,
    %   the properties of inv(A). A is singular if cond(A) is Infinite, and ill-conditioned if it is too large
    %   (Large condition numbers indicate a nearly singular matrix). A is badly conditioned if RCOND(A)
    %   is near EPS (1e-016).      
    if option == 0
        w = A\b;        % Regular solution                
    else
        % This option can be used if the previous solution is not reliable (if the inverse was found, but is not reliable)
        w = pinv(A)*b;  % Solution based on the Moore-Penrose pseudoinverse of the A matrix (pinv).
                         % If A is square and not singular, then pinv(A) is an expensive way to compute inv(A).
                         % If A is square and singular, then inv(A) does not exist. In this case, pinv(A) has some of, but not all, the properties of inv(A).
                         % Therefore, this solution is adequate only in this last case.
    end
    
end % if (nargin == 6 || nargin < 4)

