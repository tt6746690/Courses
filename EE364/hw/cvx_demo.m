% http://cvxr.com/news/2014/02/cvx-demo-video/

cvx_begin
    variables x y
    y >= 0
    minimize((x+y+3)^2)
cvx_end

% converted to sdp problem, then solve

x, y
cvx_status
cvx_optval

%   keep track `curvature` of expression
%

cvx_begin
    variables x y z;
    x + y == z                % {real affine} == {real affine}
    (x+(2*y))^2 <= min(z,2)   % {convex} <= {concave}
    % x+y <= max(x,2)         % Invalid constraint: {real affine} <= {convex}
                              %     violates DCP
    minimize(max(x,y))        % convex objective
cvx_end

% vector valued optimization 

A = randn(100,30);
b = randn(100,1);

cvx_begin
    variable x(30)
    x >= 0                    % element-wise, (scalar broadcast to vector)
    sum(x) == 1               % `sum` overloaded to accept a `cvx.variable`
    minimize(norm(A*x-b))
cvx_end


% max correlation of two random variables

cvx_begin
    variable C(3,3) symmetric
    diag(C) == 1
    C(1,2) == 0.6
    C(2,3) == -0.30
    C == semidefinite(3)       % C >= 0
    maximize(C(1,3))
cvx_end

% correlation matrix with specified correlation
%       and which maximizes the 1,3 entry
%
% C =
% 
%     1.0000    0.6000    0.5832
%     0.6000    1.0000   -0.3000
%     0.5832   -0.3000    1.0000


% Dual variable
%
% a LP problem
[A,b,c] = deal(magic(4),rand(4,1),rand(4,1));
n = size(A,2);
cvx_begin
    variable x(n);
    dual variable y;
    minimize( c' * x );
    subject to
        y : A * x <= b;
cvx_end

% complementary slackness 
y .* (b-A*x)


% expression holder 
%       z is expression holder ... not computed numerically
%
%       variables x y
%       z = 2 * x - y;
%       square( z ) <= 3;
%       quad_over_lin( x, z ) <= 1;
%
%       x is expression holder
%       
%       variable u(9);
%       expression x(10);
%       x(1) = 1;
%       for k = 1 : 9,
%           x(k+1) = sqrt( x(k) + u(k) );
%       end
%
