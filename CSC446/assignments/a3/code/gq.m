% `n`-point Gaussian Quadrature 
%       given function `f` with lower/upper limit `a`/`b`
function int = gq(f,a,b,order)
    assert(any(order == [5]), 'only support {5}-point gauss quadrature');
    
    gq5_points = [
        0
        -1/3*sqrt(5-2*sqrt(10/7))
        +1/3*sqrt(5-2*sqrt(10/7))
        -1/3*sqrt(5+2*sqrt(10/7))
        +1/3*sqrt(5+2*sqrt(10/7))
    ];

    gq5_weights = [
        128/225
        (322+13*sqrt(70)) / 900
        (322+13*sqrt(70)) / 900
        (322-13*sqrt(70)) / 900
        (322-13*sqrt(70)) / 900
    ];

    % over [-1.1]
    fx = arrayfun(@(x,w) w*f(x*(b-a)/2+(a+b)/2), ...
        gq5_points, gq5_weights);
    int = (b-a)/2 * sum(fx);
end

