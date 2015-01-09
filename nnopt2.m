% ===================================================================================================
% nnopt algorithm - Version 4.0
% ------------------------------
% Written by Oscar Fontenla-Romero, 06.05.2013 @ LIDIA,UDC - A Coru�a, Spain
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
% [A,b,w] = nnopt2(x,d,finv,fderiv,option,A,b)
%
% Parameters of the function:
% --------------------------
% x      : inputs of the network (size: I x S).
% d      : desired outputs of the network (size: 1 x S).
% finv   : inverse of the neural function.
% fderiv : derivative of the neural function
% option : flag (value = 0 or OTHER INTEGER VALUE) for the type of method used to obtain the solution (it is optional, default value = 0).
%          For classification problems is recomendable the use of option = 1 as in this kind of problems can appear ill-conditioned matrices.
% A      : Coefficient matrix of the system A*w = b in the previous training (size: I+1 x I+1). Optional. It is used for incremental learning.
% b      : Right hand side values of the system A*w = b in the previous training (size: 1 x I+1). Optional. It is used for incremental learning.
%  
% Note: The A and b parameters must be zeros (for all the components) in the first use of this
%       function for incremental learning. If A and b are not supplied the
%       function initialize both to zeros values.
% 
% Returns:
% -------
% A : Coefficient matrix of the system A*w = b (size: I+1 x I+1).
% b : Right hand side values of the system A*w = b (size: 1 x I+1).
% w : Weight vector (size: I+1 x 1).
%     The 1st element is the bias (wj0).
%
% ===================================================================================================
%  
% Algorithm: Solution of the linear system of equations (A*w = b) given by the following equations:
%  
%   S                                    I    S                                              S
%   _                                    _    _                                              _
%  \                    _               \    \                           _                  \    -1
%  /_  x_ps * df_j ^ 2 (d_js) * w_j0 +  /_ ( /_  x_is * x_ps * df_j ^ 2 (d_js) ) * w_ji  =  /_  f_j (y_js) * x_ps * df_j ^ 2 (d_js)    (eqn 1)
%  s=1                                  i=1  s=1                                            s=1
%
%
%   S                             I    S                                       S
%   _                             _    _                                       _
%  \             _               \    \                    _                  \    -1
%  /_  df_j ^ 2 (d_js) * w_j0 +  /_ ( /_  x_is * df_j ^ 2 (d_js) ) * w_ji  =  /_  f_j (y_js) * df_j ^ 2 (d_js)                         (eqn 2)
%  s=1                           i=1  s=1                                     s=1
%        
%  where: 
%
%   S : Number of samples.
%   I : Number of inputs.
%   j : the jth output (this algorithm is applied independently to each output, j = 1,...,J).
%   x_ps : x is the input vector and p = 1,..,I and s = 1,..,S.
%   y_js : y is the output of the network.
%   d_js : d is desired output.
%
%    -1
%   f_j : inverse of the jth neural function. 
%
%   _       -1
%   d_js = f_j (d_js).
%                  
%   df_j ^ 2 : square of the derivative of the jth neural function.
%
% ===================================================================================================

function [A,b,w] = nnopt2(x,d,finv,fderiv,option,A,b)

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
    % Number of inputs (I) and data points (S)
    [I,S] = size(x);    
    % The bias is included as the first input (first row)
    x = [ones(1,S); x];    
    % Inverse of the neural function
    f_d = feval(finv,d); %feval?????????????????????????;d:desired outputs of the network (size: 1 x S).    
    % As default the SVD solution is not used
    SVD_sol = 0;

    % If matrix A and b are not supplied then not incremental learning is used for A and b.    
    % System of linear equations: A*w = b    
    if (nargin ~= 7)
        % If the number of data points (S) is larger than the number of
        % variables (I) then the regular batch solution is solved as the size of
        % the system of linear equations is (I+1)x(I+1). Otherwise a SVD
        % can be used to invert the size of the system to SxS.
        if (S>=I) 
            df2 = feval(fderiv,f_d).^2; % Square of the derivate of the neural function            
            b = (f_d.*df2)*x';          % b vector for the system of equetions
            f_d = ones(I+1,1)*f_d;
            df2 = ones(I+1,1)*df2;
            A = (df2.*x)*x';            % A matrix for the system of equations                       
        else % SVD solution to invert the size of the system of linear equations
            derf = feval(fderiv,f_d);   % Derivate of the neural function
            F = diag(derf);             % Diagonal matrix
            [U,S,V] = svd(x*F,'econ');  
            Z = V*S;
            A = Z'*Z;                        
            b = f_d*F*Z;
            SVD_sol = 1;                 
        end
    else % This is used for incremental learning.                     
        df2 = feval(fderiv,f_d).^2;   % Square of the derivate of the neural function                           
        b = b + (f_d.*df2)*x';        % b vector for the system of equetions
        f_d = repmat(1,I+1,1)*f_d;
        df2 = repmat(1,I+1,1)*df2;
        A = A + (df2.*x)*x';        % A matrix for the system of equations
     end

    % If the option flag is not supplied then is determined using the
    % condition number and the reciprocal condition estimator of the matrix A
    if ( (nargin == 4) || isempty(option) )                       
        % If the matrix A is near to singular
        if ( (cond(A)<1e+016) && (rcond(A)>1e-015) )        
            option = 0;  % Regular solution
        else
            option = 1;  % Solution based on the Moore-Penrose pseudoinverse
        end
    end        
    
    % Solution to the equation A*w = b:    
    % - If A is square and not singular then the best solution is to use the inverse of A.
    % - If A is square and singular, then inv(A) does not exist. In this case, pinv(A) has some of, but not all,
    %   the properties of inv(A). A is singular if cond(A) is Infinite, and ill-conditioned if it is too large
    %   (Large condition numbers indicate a nearly singular matrix). A is badly conditioned if RCOND(A)
    %   is near EPS (1e-016).      
    if option == 0
        w = A\b';        % Regular solution                
    else
        % This option can be used if the previous solution is not reliable (if the inverse was found, but is not reliable)
        w = pinv(A)*b';  % Solution based on the Moore-Penrose pseudoinverse of the A matrix (pinv).
                         % If A is square and not singular, then pinv(A) is an expensive way to compute inv(A).
                         % If A is square and singular, then inv(A) does not exist. In this case, pinv(A) has some of, but not all, the properties of inv(A).
                         % Therefore, this solution is adequate only in this last case.
    end
    if (SVD_sol == 1)    % If the SVD solution was used
        w = U*w;
    end
end % if (nargin == 6 || nargin < 4)

