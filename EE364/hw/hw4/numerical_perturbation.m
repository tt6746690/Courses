Q = [1 -1/2; -1/2 2];
f = [-1 0]';
A = [1 2; 1 -4; 5 76];
b = [-2;-3;1];

% (a) solve the QP
cvx_begin quiet
    variable x(2);
    dual variables lambda;
    minimize quad_form(x,Q) + transpose(f)*x;
        lambda : A*x <= b;
cvx_end

pstar = cvx_optval;

% lambda =

%     1.8994            % pstar more sensitive to perturbation
%     3.4684            % pstar most sensitive to perturbation
%     0.0931            % less sensitive


% verify KKT 
assert(all( A*x <= b ));
assert(all( lambda >= 0 ));
assert(all( (abs((A*x-b).*lambda)) <= [1e-5;1e-5;1e-5] ));
assert(all( 2*Q*x+f+A'*lambda <= [1e-4;1e-4] ));

% (b) perturbed QP
valueset = [0 -0.1 0.1];
table = [];

for i = 1:3
    for j = 1:3
        delta1 = valueset(i);
        delta2 = valueset(j);
        cvx_begin quiet
            variable x(2);
            minimize quad_form(x,Q) + transpose(f)*x;
                A*x <= b+[delta1;delta2;0];
        cvx_end
        p_pred = pstar - lambda(1)*delta1 - lambda(2)*delta2;
        p_exact = cvx_optval;
        assert(p_exact >= p_pred);
        table = [table; delta1 delta2 p_pred p_exact];
    end
end

% table =

%    delta1    delta2    p_pred   p_exact
%
%         0         0    8.2222    8.2222
%         0   -0.1000    8.5691    8.7064   % notice here p_pred/p_exact larger than 
%         0    0.1000    7.8754    7.9800
%   -0.1000         0    8.4122    8.5650   % here, because lambda(1) < lambda(2)
%   -0.1000   -0.1000    8.7590    8.8156
%   -0.1000    0.1000    8.0653    8.3189
%    0.1000         0    8.0323    8.2222
%    0.1000   -0.1000    8.3791    8.7064
%    0.1000    0.1000    7.6854    7.7515


