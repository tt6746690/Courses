clear all;
global m grid;

logit = @(x) log(x/(1-x));
abs_logit = @(x) abs(logit(x));
abs_logit_scaled = @(x) 0.1*abs(logit(x));

% ms = [9];
ms = [9 19 39 79 159 319 639];

fprintf('m\tmax error\tratio\n');
for iter = 1:size(ms,2)
    m = ms(iter);
    h = 1/(m+1);
    grid = 0:h:1;
    grid = error_equidistribution(abs_logit_scaled, grid);

    [A, b] = assembly();
    c = A \ b;
    
    e = arrayfun(@(i) abs(y(grid(i+1)) - (c(i)+1)), ...
        1:m);
    max_e = max(e);
  
    plot(grid(2:end-1), arrayfun(@(i) y(grid(i+1)), 1:m), '-', ...
         grid(2:end-1), arrayfun(@(i) (c(i)+1), 1:m), '--');

    ratio = 0;
    if iter ~= 1
        ratio = max_e / pre_max_e;
    end
    fprintf('%d\t%.10f\t%.10f\n', m, max_e, ratio);
    pre_max_e = max_e;
end


function [A, b] = assembly()
    global m grid;

    A = sparse(m,m);
    b = zeros(m,1);

    % formula over [x_{i}, x_{i+1}] for lhs/rhs
    %       note `i` is 0-indexed
    akl = @(i,k,l) ...
        gq(@(x) Bp(k,x)*Bp(l,x) + (10^4)*B(k,x)*B(l,x), ...
            grid(i      +1), ...
            grid(i+1    +1), 5);

    bk = @(i,k) ...
        -(10^4)* gq(@(x) B(k,x), ...
            grid(i      +1), ...
            grid(i+1    +1), 5);

    for k = 1:m
        for l = 1:m
            if l == k
                A(k,l) = akl(k-1,k,l) + akl(k,k,l);
            elseif l == k-1
                A(k,l) = akl(l,k,l);
            elseif l == k+1
                A(k,l) = akl(k,k,l);
            end
        end
        b(k) = bk(k-1,k) + bk(k,k);
    end
end


% basis B_k at `x`
%       where `k` is 0-indexed
function Bx = B(k,x)
    global grid;

    l = grid(k-1    +1);
    c = grid(k      +1);
    r = grid(k+1    +1);

    if x >= l && x <= c
        Bx = (x-l)/(c-l);
    elseif x >= c && x <= r
        Bx = (r-x)/(r-c);
    else
        warning('B: out of support');
    end 
end


% basis B_k' at `x`
%       where `k` is 0-indexed
function Bx = Bp(k,x)
    global grid;

    l = grid(k-1    +1);
    c = grid(k      +1);
    r = grid(k+1    +1);

    if x >= l && x <= c
        Bx = 1/(c-l);
    elseif x >= c && x <= r
        Bx = 1/(c-r);
    else
        warning('B: out of support');
    end 
end

function yx = y(x)
    c_1 = (1-exp(-100))/(exp(100)-exp(-100));
    c_2 = (exp(100)-1)/(exp(100)-exp(-100));
    yx = c_1*exp(100*x) + c_2*exp(-100*x);
end

function yx = yp(x)
    yx = (100*exp(-100*x)*(exp(200*x)-exp(100)))/(exp(100)+1);
end

function yx = ypp(x)
    yx = (10000*exp(-100*x)*(exp(200*x)+exp(100)))/(exp(100)+1);
end

function phix = varphi0(x)
    phix = 1;
end


% Given monitor function that approximates erorr
%       and original grid, returns a new grid with equal error distribution
%       reference: https://www.math.uci.edu/~chenlong/226/Ch4AFEM.pdf
function xs = error_equidistribution(M, xs)
    cdf = arrayfun(@(i) gq(M, xs(i), xs(i+1), 5), 1:(max(size(xs))-1));
    cdf = [0, cumsum(cdf)];
    cdf = cdf/cdf(end);
    ys = 0:1/(length(xs)-1):1;
    [cdf, index] = unique(cdf);
    xs = interp1(cdf,xs(index),ys);
end

function explore_monitor()
    xs1 = 0:1/100:1;
    logit = @(x) log(x/(1-x));
    abs_logit = @(x) abs(logit(x));
    xs2 = error_equidistribution(abs_logit, xs1);
    plot(xs1, arrayfun(@y,xs1), ...
     xs1, arrayfun(abs_logit,xs1), ...
     xs2, zeros(size(xs2)), 'o');
end