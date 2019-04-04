clear all;
global m;

% ms = [9];
ms = [9 19 39 79 159 319 639];

fprintf('m\tmax error\tratio\n');
for iter = 1:size(ms,2)
    m = ms(iter);
    h = 1/(m+1);

    [A, b] = assembly();
    c = A \ b;

    actual = zeros(m,m);
    expected = zeros(m,m);
    for p = 1:m^2
        [i,j] = p2ij(p);
        actual(i,j) = c(p);
        expected(i,j) = u(i*h,j*h);
    end

    e = arrayfun(@(x) abs(x), expected-actual);
    max_e = max(e,[],'all');

    % [X,Y] = meshgrid(h:h:1-h);
    % mesh(X,Y,e);
        
    ratio = 0;
    if iter ~= 1
        ratio = max_e / pre_max_e;
    end
    fprintf('%d\t%.10f\t%.10f\n', m, max_e, ratio);
    pre_max_e = max_e;
end

function [A, b] = assembly()
    global m;
    h = 1/(m+1);

    % construct `A`

    % 3 integral values for `a_{kl}`
    dfull   = zeros(m^2,1) + 8/3;
    dside   = zeros(m^2,1) - 1/3;
    dcorner = zeros(m^2,1) - 1/3;

    % pad zeros at certain locations -> block diagonal
    fillzerosat = m:m:m^2;
    b1 = dside;
    b1(fillzerosat) = 0;
    b2 = dcorner;
    b2(fillzerosat) = 0;

    b = [-m-1,-m,-m+1,    -1,0,1,        m-1,m,m+1];
    B = [b2,dside,b2,   b1,dfull,b1,   b2,dside,b2];
    B(:,3) = flip(B(:,1));
    B(:,6) = flip(B(:,4));
    B(:,9) = flip(B(:,7));
    A = spdiags(B,b,m^2,m^2);

    % construct `b`

    b = zeros(m^2,1);

    % part of (half) the function for `bkl`
    bkl_ = @(k) ...
        gq(@(x) 32*x*(1-x)*hat(k,x), (k-1)*h, k*h, 5) + ...
        gq(@(x) 32*x*(1-x)*hat(k,x), k*h, (k+1)*h, 5);

    % evaluate the integral for `b_{k,l}`
    bkl = @(k,l) ...
        h*(bkl_(k) + bkl_(l));

    for j = 1:m
        for i = 1:m
            p = ij2p(i,j);
            b(p) = bkl(i,j);
        end
    end
end

function [i,j] = p2ij(p)
    global m;
    j = floor((p-1) / m) + 1;
    i = mod(p-1, m) + 1;
end
function p = ij2p(i,j)
    global m;
    p = i + (j-1)*m;
end

function fx = hat(k, x)
    global m;
    h = 1/(m+1);

    l = (k-1)*h;
    c = k*h;
    r = (k+1)*h;

    if x >= l && x <= c
        fx = 1-k+x/h;
    elseif x >= c && x <= r
        fx = 1+k-x/h;
    else
        % warning('hat function should not reach here ...');
        fx = 0;
    end
end

function fxy = f(x,y)
    fxy = 32*x*(1-x) + 32*y*(1-y);
end

function uxy = u(x,y)
    uxy = 16*x*(1-x)*y*(1-y);
end
