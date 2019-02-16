global m n od;

od = 1;
[ij2p, p2ij, n_point] = make_conversion_table();

fprintf('n\torder\terror\n');
for i = [9 19 39 79]
    m = i; n = i;
    Dx = 1/(n+1);
    for approx_order = [1 2]
        od = approx_order;
        [A, rhs] = make_Arhs();
        solution = A \ rhs;
        solution = v2m(solution);
        exact_solution = maketildeu();
        e = max(arrayfun(@(x) abs(x), ...
            solution-exact_solution), [], 'all');
        fprintf('%d\t%d\t%.10f\n', n, od, e);
%         plot_err(exact_solution-solution);
%         diff = exact_solution-solution;
%         [X,Y] = meshgrid(-1:Dx:-1+n*Dx);
%         mesh(X,Y,arrayfun(@(x) abs(x), diff(1:(n+1),1:(n+1))));
%         saveas(gcf,strcat('q6_corder_od_',num2str(od),'.png'));
    end
end


% plot error function at grid points
function plot_err(V)
    global n;
    Dx = 1/(n+1);
    [X,Y] = meshgrid(-1:Dx:1);
    mesh(X,Y,arrayfun(@(x) abs(x), V));
end



function [A rhs] = make_Arhs()
    global n od;

    Dx = 1/(n+1); Dy = Dx;
    [ij2p, p2ij, n_point] = make_conversion_table();

    A = sparse(n_point, n_point);
    rhs = zeros(n_point, 1);

    for p = 1:n_point
        i = p2ij(p,1);
        j = p2ij(p,2);
        x1 = -1 + Dx*(i-1);
        y1 = -1 + Dy*(j-1);

        find_tau_y = @(x,y) (sqrt(1-x^2) - abs(y))/Dy;
        find_tau_x = @(x,y) (sqrt(1-y^2) - abs(x))/Dx;

        % x-direction
        if ij2p(i+1,j)==0 || ij2p(i-1,j)==0
            % near-boundary point
            P = ij2p(i,j); 
            Q = 0; V = 0;
            if ij2p(i+1,j)==0
                Q = ij2p(i-1,j);
                V = ij2p(i-2,j);
            else
                Q = ij2p(i+1,j);
                V = ij2p(i+2,j);
            end
            tau = find_tau_x(x1,y1);
            assert(tau < 1.001 && tau > -0.001);
            if od == 1
                A(p,Q) = 2/(tau+1) * 1/Dx^2;
                A(p,P) = A(p,P) -2/tau * 1/Dy^2;
                rhs(p) = rhs(p) - 2/(tau*(tau+1)) * 1/Dx^2;
            else
                A(p,V) = (tau-1)/(tau+2) * 1/Dx^2;
                A(p,Q) = 2*(2-tau)/(tau+1) * 1/Dx^2;
                A(p,P) = A(p,P) -(3-tau)/tau * 1/Dy^2;
                rhs(p) = rhs(p) - 6/(tau*(tau+1)*(tau+2)) * 1/Dx^2;
            end
        else
            % interior point
            A(p,p          ) = A(p,p) -2/Dy^2;
            A(p,ij2p(i-1,j)) = 1/Dx^2;
            A(p,ij2p(i+1,j)) = 1/Dx^2;
        end

        % y-direction
        if ij2p(i,j+1)==0 || ij2p(i,j-1)==0
            % near boundary point
            P = ij2p(i,j);
            Q = 0; V = 0;
            if ij2p(i,j+1)==0
                Q = ij2p(i,j-1);
                V = ij2p(i,j-2);
            else
                Q = ij2p(i,j+1);
                V = ij2p(i,j+2);
            end
            tau = find_tau_y(x1,y1);
            assert(tau < 1.001 && tau > -0.001);
            if od == 1
                A(p,Q) = 2/(tau+1) * 1/Dy^2;
                A(p,P) = A(p,P) -2/tau * 1/Dy^2;
                rhs(p) = rhs(p) - 2/(tau*(tau+1)) * 1/Dy^2;
            else
                A(p,V) = (tau-1)/(tau+2) * 1/Dy^2;
                A(p,Q) = 2*(2-tau)/(tau+1) * 1/Dy^2;
                A(p,P) = A(p,P) -(3-tau)/tau * 1/Dy^2;
                rhs(p) = rhs(p) - 6/(tau*(tau+1)*(tau+2)) * 1/Dy^2;
            end
        else
            % interior point
            A(p,p          ) = A(p,p) -2/Dy^2;
            A(p,ij2p(i,j-1)) = 1/Dy^2;
            A(p,ij2p(i,j+1)) = 1/Dy^2;
        end
        rhs(p) = rhs(p) + f(x1,y1);
    end
end


function tildeu = maketildeu()
    global n;
    Dx = 1/(n+1); Dy = Dx;
    tildeu = zeros(2*n+3, 2*n+3);
    [ij2p, p2ij, n_point] = make_conversion_table();
    for p = 1:n_point
        i = p2ij(p,1);
        j = p2ij(p,2);
        x = -1 + Dx*(i-1);
        y = -1 + Dy*(j-1);
        tildeu(i,j) = tu(x,y);
    end
end


function mat = v2m(v)
    global n;
    mat = zeros(2*n+3,2*n+3);
    [~,p2ij,n_point] = make_conversion_table();
    for p = 1:n_point
        i = p2ij(p,1);
        j = p2ij(p,2);
        mat(i,j) = v(p);
    end
end


% construct matrix `ij2p` and `p2ij` s.t.
%   ij2p(i,j)  is  p
%   p2ij(p,:)  is  [i j]
%       where `i,j = 1,2,...,n+2` are grid indices
%       and `p = 1,2,...,n_gridpoint_inside_Omega` are indices to `u,b,f`

function [ij2p, p2ij, n_point] = make_conversion_table()
    global n;
    Dx = 1/(n+1); Dy = Dx;
    ij2p = zeros(2*n+3, 2*n+3);
    nnz = 1;
    for j = 1:(2*n+3)
        y = -1 + Dy*(j-1);
        for i = 1:(2*n+3)
            x = -1 + Dx*(i-1);
            if x^2+y^2 < 1
                ij2p(i, j) = nnz;
                nnz = nnz+1;
            end
        end
    end
    n_point = nnz-1;
    p2ij = zeros(n_point, 2);
    nnz = 1;
    for j = 1:(2*n+3)
        y = -1 + Dy*(j-1);
        for i = 1:(2*n+3)
            x = -1 + Dx*(i-1);
            if x^2+y^2 < 1
                p2ij(nnz,:) = [i j];
                nnz = nnz + 1;
            end
        end
    end
    % sanity check
    for k = 1:(nnz-1)
        assert(ij2p(p2ij(k,1), p2ij(k,2)) == k);
    end
end


% evaluate function `f` at grid location `(x,y)`
function fxy = f(x, y)
    fxy = 16*(x^2+y^2);
end

% ground truth value for `u`
function uxy = tu(x, y)
    uxy = (x^2+y^2)^2;
end

function y = sqrd(x, y)
    y = x^2+y^2;
end

