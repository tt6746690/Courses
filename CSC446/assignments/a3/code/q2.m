clear all;
global m;

ms = [10,20,40,80,160,320,640];
% ms = [10];

fprintf('m\tmax error\tratio\n');
for iter = 1:size(ms,2)
    m=ms(iter);
    h=1/m;

    [S, P] = getSP();
    [A, b] = assembly();
    c = A \ b;

    nodes = zeros(1, m);   % at x_{i=1} to 1
    for k = -2:(m-1)
        Bk_support = S(k+3,:);
        for i = Bk_support(1):Bk_support(2)
            nodes(i+1) = nodes(i+1) + c(k+3)*B(k, P(k+3,i+1), (i+1)*h);
        end
    end

    e = arrayfun(@(i) abs(y(i*h) - (nodes(i)+varphi0(i*h))), ...
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
    N = m+2;

    A = sparse(N,N);
    b = zeros(N,1);

    % k = Bk; l = Bl;
    % a_{k,l} on [x_i, x_{i+1}]
    %       contributed by B_{k}^Pk, B_{l}^Pl
    akl = @(i,Bk,Pk,Bl,Pl) ...
        gq(@(x) (Bp(Bl,Pl,x)*Bp(Bk,Pk,x) + B(Bl,Pl,x)*B(Bk,Pk,x)), ...
            i*h, (i+1)*h, 5);

    % b_k on [x_i, x_{i+1}]
    %       contributed by B_{k}^Pk, B_{l}^Pl
    bk = @(i,Bk,Pk) ...
        gq(@(x) (f(x) - varphi0(x))*B(Bk,Pk,x), ...
            i*h, (i+1)*h, 5);

    [S, P] = getSP();

    for k = -2:(m-1)
        for l = -2:(m-1)
            Bk_support = S(k+3,:);
            Bl_support = S(l+3,:);
            support = [
                max(Bk_support(1),Bl_support(1)), ...
                min(Bk_support(2),Bl_support(2)), ...
            ];

            % support does not overlap, so a_{k,l} = 0
            if support(1)>support(2)
                continue;
            end

            A(k+3,l+3) = 0;
            for i = support(1):support(2)
                A(k+3,l+3) = A(k+3,l+3) + akl(i,k,P(k+3,i+1),l,P(l+3,i+1));
            end
        end

        Bk_support = S(k+3,:);
        for i = Bk_support(1):Bk_support(2)
            b(k+3) = b(k+3) + bk(i,k,P(k+3,i+1));
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

function [S, P] = getSP()
    global m;
    N = m+2;

    % support (m+2)x2
    %       where [S(k+3,1), S(k+3,2)+1] is support for basis B_k
    S = zeros(N,2);
    S(1,:)   = [0, 1];      % B_{-2}
    S(2,:)   = [0, 2];      % B_{-1}
    for i = 3:(m-1)
        S(i,:) = [i-3, i];
    end
    S(m,:)   = [m-3, m-1];  % B_{m-3}
    S(m+1,:) = [m-2, m-1];  % B_{m-2}
    S(m+2,:) = [m-1, m-1];  % B_{m-1}

    % polynomial (m+2)xm
    %       where P(k+3, i+1) is {0,1,2,3}-th polynomial 
    %           for cubic B_k (k=-2,...,m-1) 
    %           on [x_i, x_{i+1}] (i=0,...,m-1)
    P = zeros(N,m);
    P(:,:) = -1;
    P(1,1:2) = [0, 1];      % B_{-2}
    P(2,1:3) = [0, 1, 2];   % B_{-1}
    for i = 3:(m-1)
        P(i,(i-2):(i+1)) = [0, 1, 2, 3];
    end
    P(m,(m-2):m)   = [0,1,2];% B_{m-3}
    P(m+1,(m-1):m) = [0,1];  % B_{m-2}
    P(m+2,m:m)     = [0];    % B_{m-1}
end


% polynomial function for `i`-th basis function `B_i`
%       and `k`-th polynomial `P_k` for `B_i`  where k \in {0,1,2,3}
function Bx = B(i, k, x)
    global m;
    h = 1/m;

    if i == -2
        switch k
        case 0
            Bx = (3)*(x/h) - (9/2)*(x/h)^2 + (7/4)*(x/h)^3;
        case 1
            Bx = (1/4)*((2*h-x)/h)^3;
        otherwise
            warning('k ~\in {0,1} for B_{-2)\n');
        end
        return;
    end

    if i == -1
        switch k
        case 0
            Bx = (3/2)*(x/h)^2 - (11/12)*(x/h)^3;
        case 1
            Bx = (-3/2) + (9/2)*(x/h) - (3)*(x/h)^2 + (7/12)*(x/h)^3;
        case 2
            Bx = (1/6)*((3*h-x)/h)^3;
        otherwise
            warning('k ~\in {0,1,2} for B_{-1)\n');
        end
        return;
    end

    if  (i == m-1 && any(k == [1 2 3])) || ...
        (i == m-2 && any(k == [2 3])) || ...
        (i == m-3 &&     k == 3)
        warning('k not right value for for B_{m-1,m-2,m-3}\n');
        return;
    end

    assert((i >= 0 && i <= m-1),'invalid basis i');
    l = i*h;
    r = (i+4)*h;

    switch k
    case 0
        Bx = (1/6)*((x-l)/h)^3;
    case 1
        Bx = (2/3)   - (2)*((x-l)/h)  + (2)*((x-l)/h)^2  - (1/2)*((x-l)/h)^3;
    case 2
        Bx = (-22/3) + (10)*((x-l)/h) - (4)*((x-l)/h)^2  + (1/2)*((x-l)/h)^3;
    case 3
        Bx = (1/6)*((r-x)/h)^3;
    otherwise
        warning('k ~\in {0,1,2,3} for B_i\n');
        return;
    end
end

% first order derivative of polynomial function for `i`-th basis function `B_i`
%       and `k`-th polynomial `P_k` for `B_i`  where k \in {0,1,2,3}
function Bx = Bp(i, k, x)
    global m;
    h = 1/m;

    if i == -2
        switch k
        case 0
            Bx = (21*x^2)/(4*h^3) - (9*x)/(h^2) + 3/h;
        case 1
            Bx = -(3*(2*h-x)^2)/(4*h^3);
        otherwise
            warning('Bp: k ~\in {0,1} for B_{-2)\n');
        end
        return;
    end

    if i == -1
        switch k
        case 0
            Bx =  -x*(11*x-12*h)/(4*h^3);
        case 1
            Bx = 9/(2*h) - (6*x)/(h^2) + (7*x^2)/(4*h^3);
        case 2
            Bx = -(3*h-x)^2/(2*h^3);
        otherwise
            warning('Bp: k ~\in {0,1,2} for B_{-1)\n');
        end
        return;
    end

    if  (i == m-1 && any(k == [1 2 3])) || ...
        (i == m-2 && any(k == [2 3])) || ...
        (i == m-3 &&     k == 3)
        warning('Bp: k not right value for for B_{m-1,m-2,m-3}\n');
        return;
    end

    assert((i >= 0 && i <= m-1),'Bp: invalid basis i');
    l = i*h;
    r = (i+4)*h;

    switch k
    case 0
        Bx =  (x-l)^2/(2*h^3);
    case 1
        Bx = -2/h + 4*(x-l)/h^2 - 3*(x-l)^2/(2*h^3);
    case 2
        Bx = 10/h - 8*(x-l)/h^2 + 3*(x-l)^2/(2*h^3);
    case 3
        Bx =  -(r-x)^2/(2*h^3);
    otherwise
        warning('Bp: k ~\in {0,1,2,3} for B_i\n');
    end
end
