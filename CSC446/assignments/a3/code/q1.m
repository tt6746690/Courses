clear all;
global m;

ms = [10,20,40,80,160,320,640];
% ms = [10];

fprintf('m\tmax error\tratio\n');
for iter = 1:size(ms,2)
    m=ms(iter);
    h=1/m;

    [A, b] = assembly();
    c = A \ b;

    e = arrayfun(@(i) abs(y(i*h) - (c(i)+varphi0(i*h))), ...
        1:m);
    max_e = max(e);
    
    xs = arrayfun(@(i)i*h,1:m);
    plot(xs, arrayfun(@(i) y(i*h), 1:m), '-', ...
         xs, arrayfun(@(i) (c(i)+varphi0(i*h)), 1:m), '--');

    ratio = 0;
    if iter ~= 1
        ratio = max_e / pre_max_e;
    end
    fprintf('%d\t%.10f\t%.10f\n', m, max_e, ratio);
    pre_max_e = max_e;
end

function [A, b] = assembly()
    global m;
    
    h = 1/m;
    A = sparse(m,m);
    b = zeros(m,1);

    for k = 1:m
        if k ~= 1
            A(k,k-1) = (h^2-6)/(6*h);
        end
        if k ~= m
            A(k,k) = 2*(h^2+3)/(3*h);
            A(k,k+1) = (h^2-6)/(6*h);
            b(k) = gq(@(x) (f(x)-varphi0(x)).*(1-k+x./h), (k-1)*h, k*h, 5) ...
                +  gq(@(x) (f(x)-varphi0(x)).*(1+k-x./h), k*h, (k+1)*h, 5);
        else
            A(k,k) =   (h^2+3)/(3*h);
            b(k) = gq(@(x) (f(x)-varphi0(x)).*(1-k+x./h), (k-1)*h, k*h, 5);
        end
    end
end


function fx = f(x)
    fx = (pi^2+1)*sin(pi*x);
end

function yx = y(x)
    yx = 1/2*(exp(x)+exp(-x)) + sin(pi*x);
end

function varphi0x = varphi0(x)
    varphi0x = (1/2*(exp(1)-exp(-1)) - pi)*x + 1;
end

function varphi0p = varphi0prime()
    varphi0p = (1/2*(exp(1)-exp(-1)) - pi);
end
