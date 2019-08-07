
% (a)
cvx_begin quiet
    variable x(1);
    minimize x^2+1;
        (x-2)*(x-4) <= 0;
cvx_end

pstar = cvx_optval;
assert(abs(cvx_optval-5) <= 1e-5);
assert(abs(x-2) <= 1e-5);

% (b)
%   plot a few lagrangian,
figure;
xs = 0:0.1:5;
f0 = @(x) x^2+1;
L = @(x,lambda) x^2+1+(x-2)*(x-4)*lambda;
plot(xs,arrayfun(f0,xs)); hold on;
plot(xs,arrayfun(@(x) L(x,0),xs),'Color','r'); hold on;
plot(xs,arrayfun(@(x) L(x,1),xs),'Color','k'); hold on;
plot(xs,arrayfun(@(x) L(x,2),xs),'Color','k'); hold on;
plot(xs,arrayfun(@(x) L(x,3),xs),'Color','k'); hold on;
hold off;

% and the dual function 
xs = (-1+0.1):0.1:4;
g = @(lambda) -9*lambda^2/(1+lambda) + 1 + 8*lambda;
figure;
plot(xs,arrayfun(g,xs));

% verify lower bound property 
% (c) the dual 
cvx_begin quiet
    variable lambda(1);
    maximize -quad_over_lin(3*lambda,1+lambda) + 1 + 8*lambda;
        lambda >= 0;
cvx_end

assert(abs(lambda-2)<=1e-5);
assert(pstar>=g(lambda));


% (d) sensitivity analysis

h = 0.1
us = -1:h:1;
pstars = zeros(size(us));

for i = 1:size(us,2);
    cvx_begin quiet
        variable x(1);
        minimize x^2+1;
            (x-2)*(x-4) <= us(i);
    cvx_end
    pstars(i) = cvx_optval;
end 

% Verify \partial p*(0) / \partial u = - \lambda^*
%   graphically
plot(us,pstars); hold on;
plot(us,arrayfun(@(u) pstar-lambda*u,us)); hold off;
title('Plot of optimal value as a function of perturbation `u`');
% numerically, with 5-point midpoint estimation of gradient
assert( ...
    abs( ...
        1/(12*h)*( pstars(9) - 8*pstars(10) + 8*pstars(12) - pstars(13) ) ...
        - (-2) ...
    ) <= 1e-4 ...
);
