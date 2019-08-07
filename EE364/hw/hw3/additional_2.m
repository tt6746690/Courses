

cvx_begin quiet
    variables x y z;
    
    % (a)
    % norm([x+2*y, x-y]) == 0;
    % Invalid constraint: {convex} == {constant}
    x+2*y == 0;
    x-y == 0;
    
    % (b)
    % `square` can only accept affine arguments, since its cvx but not inc
    % square(square(x+y)) <= x-y;
    % Illegal operation: square( {convex} ).
    square_pos(square(x+y)) <= x-y;
    
    % (c)
    % 1/x not convex on all of R
    % 1/x + 1/y <= 1;x >= 0; y >= 0
    % Cannot perform the operation: {positive constant} ./ {real affine}
    inv_pos(x) + inv_pos(y) <= 1;
    
    % (d)
    % `norm` can only accept affine arguments, since its cvx but not desc
    % norm([max(x,1), max(y,2)]) <= 3*x + y
    % Cannot perform the operation norm( {convex}, 2 )
    %
    % Fix by introducing new variables
    variables u v;
    norm([u,v]) <= 3*x+y;
    max(x,1) <= u;
    max(y,2) <= v;
    
    % (e)
    % xy not concave for x,y>=0
    % x*y >= 1; x >= 0; y >= 0;
    % Invalid quadratic form(s): not a square.
    % 2 possible ways to fix 
    x >= inv_pos(y); x >= 0;
    % or LMI representation 
    [x 1; 1 y] == semidefinite(2);
    
    % (f)
    % `y` concave, cannot divide concave function 
    % (x+y)^2 / sqrt(y) <= x-y+5
    % Cannot perform the operation: {convex} ./ {concave}
    quad_over_lin(x+y,sqrt(y)) <= x-y+5;
    
    % (g)
    % ^3 not convex for x < 0 ...
    % x^3+y^3 <= 1; x>=0; y>=0;
    % Illegal operation: {real affine} .^ {3}
    % (Consider POW_P, POW_POS, or POW_ABS instead.)
    pow_pos(x,3) + pow_pos(y,3) <= 1;
    
    % (h)
    % xy not concave (arg to sqrt must be concave)
    % x+z <= 1+sqrt(x*y-z^2); x >= 0; y >= 0;
    % Invalid quadratic form(s): not a square.
    % fix by rewriting expression 
    x+z <= 1 + geo_mean([x-quad_over_lin(z,y), y]);
   
cvx_end