% `n`-point Gaussian Quadrature in 2 dimension
%       given function `f` with lower/upper limit `a`/`b`
function int = cubature2(f,a,b,c,d,order)
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

    gq5_weights2d = gq5_weights.*gq5_weights';

    fxs = zeros(5,5);
    for i = 1:5
        for j = 1:5
            x = gq5_points(i);
            y = gq5_points(j);
            w = gq5_weights2d(i,j);
            fxs(i,j) = w*f(x*(b-a)/2+(a+b)/2, y*(d-c)/2+(d+c)/2);
        end
    end

    int = (b-a)/2 * (d-c)/2 * sum(fxs,'all');
end

