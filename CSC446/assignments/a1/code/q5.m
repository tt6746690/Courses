global m n;
global od;  % order of approximation to boundary derivative

fprintf('n\torder\terror\n');
for i = [9 19 39 79]
    m = i; n = i;
    Dx = 1/(m+1);
    for approx_order = [1 2]
        od = approx_order;
        solution = poisson_solver();
        solution = v2m(solution);
        exact_solution = maketildeu();
        for j = 1:n
            solution(m+1,j) = -0.25*Dx + solution(m, j);
        end
        e = max(arrayfun(@(x) abs(x), solution-exact_solution), [], 'all');
        fprintf('%d\t%d\t%.10f\n', n, od, e);
        plot_err(exact_solution, solution);
        saveas(gcf,strcat('q5_od_',num2str(od),'.png'));
    end
end


% plot error function at grid points
function plot_err(expected, actual)
    global m n;
    Dx = 1/(m+1);
    Dy = 1/(n+1);
    [X,Y] = meshgrid(0+Dx:Dx:1-Dx, 0+Dy:Dy:1);
    mesh(X,Y,arrayfun(@(x) abs(x), actual-expected));
end

% construct and solve poisson equation over (0,1)^2
%       under dirichlet boundary condition on 3 sides
%       and Neumann boundary condition on 1 side
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
        for i = 1:(m+1)
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
                 boundary_value(Dx*(i-1), y) ...
                +boundary_value(Dx*(i+1), y) ...
                +boundary_value(x, Dy*(j-1)) ...
                +boundary_value(x, Dy*(j+1));

            % populate rhs of `Au = f + b`
            rhs(p,1) = fxy + bnd;
        end
    end
end


% construct matrix `A` on (0,1)^2
%       A    m*n x m*n
function A = makeA()
    global m n od;
    Dx = 1/(m+1);
    Dy = 1/(n+1);
    Tsupdiag = repmat(1/Dx^2, m, 1);
    Tdiag = repmat(-2./Dx^2 -2/Dy^2, m, 1);
    Tsubdiag = repmat(1/Dx^2, m, 1);
    Tfar = repmat(1/Dy^2, m, 1);
    Tsupdiag(1,1) = 0;
    Tsubdiag(m,1) = 0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Neumann boundary changes
    if od == 1
        Tdiag(m,1) = -1/Dx^2 -2/Dy^2;
    elseif od == 2
        Tdiag(m,1) = -2/(3*Dx^2) - 2/Dy^2;
        Tsubdiag(m-1,1) = 2/(3*Dx^2);
    else
        assert(false);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Tdiag = repmat(Tdiag, n, 1);
    Tsupdiag = repmat(Tsupdiag, n, 1);
    Tsubdiag = repmat(Tsubdiag, n, 1);
    Tfar = repmat(Tfar, n, 1);
    A = spdiags([Tfar Tsubdiag Tdiag Tsupdiag Tfar], [-m -1 0 1 m], m*n,m*n);
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

% evaluate `u` at dirichlet boundary on 3 sides and Neumann boundary on 1 side
%       outputs `0` for (x,y) not on boundary
function uxy = boundary_value(x, y)
    global m n od;
    Dx = 1/(m+1);
    Dy = 1/(n+1);
    
    if x == 0
        uxy = -(1 + 1/(1+y))/Dx^2;
    elseif x == 1 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Neumann boundary changes
        if od == 1
            uxy = 1/(4*Dx);
        elseif od == 2
            uxy = 1/(6*Dx);
        else
            assert(false);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    elseif y == 0
        uxy = -(1 + 1/(1+x))/Dy^2;
    elseif y == 1
        uxy = -(1/2 + 1/(1+x))/Dy^2;
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