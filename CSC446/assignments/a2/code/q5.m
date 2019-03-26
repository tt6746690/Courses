
global m;
ms = [9 19 39 79 159 319 639];
fprintf('m\terror\t\terror_ratio\n')
for n_hat = ms
    m = n_hat;
    h = 1/(m+1);
    [A b] = make_Ab();
    c = A \ b;
    e = zeros(m);
    for i = 1:m
        xi = i*h;
        e(i) = abs(y(xi) - (c(i)+1));
    end
    max_e = max(e, [], 'all');
    if m == 9
        e_norm = max_e;
    end
    fprintf('%d\t%.10f\t%.10f\n', m, max_e, max_e/e_norm);
    e_norm = max_e;
end

function [A b] = make_Ab()
    global m;
    h = 1/(m+1);
    A = sparse(m,m);
    b = zeros(m,1);
    for k = 1:m
        for l = 1:m
            if l == k
                A(k,l) = 2*(10*h^2+3) / (3*h);
            elseif l == k-1 || l == k+1
                A(k,l) = (5*h^2-3) / (3*h);
            end
        end
        b(k) = k*h^2 - 10*h;
    end
end

function yx = y(x)
    l = sqrt(10);
    c_2 = (exp(l) + 1/l^2 - 1) / (exp(l) - exp(-l));
    c_1 = 1 - c_2;
    yx = c_1*exp(l*x) + c_2*exp(-l*x) + x/l^2;
end