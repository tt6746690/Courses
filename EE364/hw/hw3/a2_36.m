
cvx_begin
    variables x y z;
    
    % sqrt(cvx) not dcp compliant
    %       since sqrt concave nondecreasing, 
    %       argument must be concave, but is convex
    % sqrt(1+4*square(x)+16*square(y)) >= 0;
    
    % dcp compliant
    min(x,log(y)) - max(y,z);
    
    % not sure why this is dcp compliant
    log(exp(2*x+3)) + exp(4*y+5);
    
cvx_end

