global m n;

fprintf('n\terror\n');
for i = [9, 19, 39, 79]
    m = i; n = i;
    solution = poisson_solver();
    exact_solution = maketildeu();
    e = max(arrayfun(@(x) abs(x), ...
        v2m(solution)-exact_solution), [], 'all');
    fprintf('%d\t%.10f\n', n, e);
end

% plot error function at grid points
function plot_err(expected, actual)
    global n;
    Dx = 1/(n+1);
    [X,Y] = meshgrid(0+Dx:Dx:1-Dx);
    mesh(X,Y,arrayfun(@(x) abs(x), actual-expected));
end


% construct and solve poisson equation over (0,1)^2
%       under dirichlet boundary condition
function sol = poisson_solver()
    A   = makeA();
    rhs = makerhs();
    sol = A \ rhs;
end


% gives exact solution to `A\tilde{u} = f+b`
%       tildeu   m x n
function tildeu = maketildeu()
    global m n;
    Dx = 1/(m+1);
    Dy = 1/(n+1);
    tildeu = zeros(m, n);
    for j = 1:n
        for i = 1:m
            tildeu(i, j) = tu(Dx*i, Dy*j);
        end
    end
end


% make right hand side to `Au = f+b`
%       rhs     m*n x 1
function rhs = makerhs()
    global m n;
    Dx = 1/(m+1);
    Dy = 1/(n+1);
    
    rhs = zeros(m*n, 1);
    for j = 1:n
        for i = 1:m
            % index to rhs
            p = ij2p(i, j);
            % x,y value 
            x = Dx*i;
            y = Dy*j;

            % `f` evaluated at (x,y)
            fxy = f(x, y);

            % determine boundary value
            bnd = ...
                -u(Dx*(i-1), y)/Dx^2 ...
                -u(Dx*(i+1), y)/Dx^2 ...
                -u(x, Dy*(j-1))/Dy^2 ...
                -u(x, Dy*(j+1))/Dy^2;

            % populate rhs of `Au = f + b`
            rhs(p,1) = fxy + bnd;
        end
    end
end

% subroutines for conversion between
%       - grid indexed matrix    m   x n
%       - flattened vectors      m*n x 1
function mat = v2m(v)
    global m n;
    mat = zeros(m, n);
    for p = 1:(m*n)
        [i, j] = p2ij(p);
        mat(i, j) = v(p,1);
    end
end
function v = m2v(mat)
    global m n;
    v = zeros(m*n, 1);
    for j = 1:n
        for i = 1:m
            p = ij2p(i,j);
            v(p,1) = mat(i, j);
        end
    end
end

% subroutines for conversion between 
%       - grid indices (i, j)
%       - corresponding index in `u,f,b` `p`
function [i, j] = p2ij(p)
    global m;
    j = floor((p-1) / m) + 1;
    i = mod(p-1, m) + 1;
end
function p = ij2p(i,j)
    global m;
    p = i + (j-1)*m;
end

% construct matrix `A` on (0,1)^2
%       A    m*n x m*n
function A = makeA()
    global m n;
    Dx = 1/(m+1);
    Dy = 1/(n+1);
    Tsupdiag = repmat(1/Dx^2, m, 1);
    Tdiag = repmat(-2./Dx^2 -2/Dy^2, m, 1);
    Tsubdiag = repmat(1/Dx^2, m, 1);
    Tfar = repmat(1/Dy^2, m, 1);
    % When m == n or m > n, spdiags takes 
    %       elements of the super-diagonal in A from the lower part of the corresponding column of B, and 
    %       elements of the sub-diagonal   in A from the upper part of the corresponding column of B.
    Tsupdiag(1,1) = 0;
    Tsubdiag(m,1) = 0;
    Tdiag = repmat(Tdiag, n, 1);
    Tsupdiag = repmat(Tsupdiag, n, 1);
    Tsubdiag = repmat(Tsubdiag, n, 1);
    Tfar = repmat(Tfar, n, 1);
    A = spdiags([Tfar Tsubdiag Tdiag Tsupdiag Tfar], [-m -1 0 1 m], m*n,m*n);
end


% evaluate `u` at dirichlet boundary
%       outputs `0` for (x,y) not on boundary
function uxy = u(x, y)
    if x == 0
        uxy = 1 + 1/(1+y);
    elseif x == 1
        uxy = 1/2 + 1/(1+y);
    elseif y == 0
        uxy = 1 + 1/(1+x);
    elseif y == 1
        uxy = 1/2 + 1/(1+x);
    else
        uxy = 0;
    end
end

% evaluate function `f` at grid location `(x,y)`
function fxy = f(x, y)
    fxy = 2/(1+x)^3 + 2/(1+y)^3;
end

% ground truth value for `u`
function uxy = tu(x, y)
    uxy = 1/(1+x) + 1/(1+y);
end