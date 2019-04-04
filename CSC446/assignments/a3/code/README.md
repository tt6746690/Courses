


    % We need second derivatives for regular basis functions given as follows 
    % \[
    %     B_i^{(2)}(x) = 
    %     \begin{cases}
    %         \dfrac{x-x_i}{h^3} & x\in [x_i,x_{i+1}] \\
    %         \dfrac{4}{h^2} - \dfrac{3(x-x_i)}{h^3} & x\in[x_{i+1}, x_{i+2}] \\
    %         -\dfrac{8}{h^2} + \dfrac{3(x-x_i)}{h^3} & x\in[x_{i+2}, x_{i+3}] \\
    %         \dfrac{x_{i+4}-x}{h^3} & x\in [x_{i+3}, x_{i+4}] \\
    %         0 & \text{otherwise} \\
    %     \end{cases}
    % \]
    % for $i=0,1,\cdots,m-1$ and second derivatives for special basis function
    % \[
    %     B_{-1}^{(2)}(x) = 
    %     \begin{cases}
    %         \dfrac{3}{h^2} - \dfrac{11}{2h^3} x & x\in[0,h] \\
    %         -\dfrac{6}{h^2} + \dfrac{7}{2h^3} x & x\in[h,2h] \\
    %         \dfrac{3h-x}{h^3} & x\in[2h,3h] \\
    %         0 & \text{otherwise} \\
    %     \end{cases}
    % \]
    % \[
    %     B_{-2}^{(2)}(x) = 
    %     \begin{cases}
    %         -\dfrac{9}{h^2} + \dfrac{21}{2h^3} x & x\in[0,h] \\
    %         \dfrac{3(2h-x)}{2h^3} & x\in[h,2h] \\
    %         0 & \text{otherwise} \\
    %     \end{cases}
    % \]



% second order derivative of polynomial function for `i`-th basis function `B_i`
%       and `k`-th polynomial `P_k` for `B_i`  where k \in {0,1,2,3}
function Bx = Bpp(i, k, x)
    global m;
    h = 1/m;

    if i == -2
        switch k
        case 0
            Bx = -(9/(h^2)) + (21/(2*h^3))*x;
        case 1
            Bx = (3*(2*h-x))/(2*h^3);
        otherwise
            warning('Bpp: k ~\in {0,1} for B_{-2)\n');
        end
        return;
    end

    if i == -1
        switch k
        case 0
            Bx =  3/(h^2) - (11/(2*h^3))*x;
        case 1
            Bx = -6/(h^2) + (7/(2*h^3))*x;
        case 2
            Bx = (3*h-x)/h^3;
        otherwise
            warning('Bpp: k ~\in {0,1,2} for B_{-1)\n');
        end
        return;
    end

    if  (i == m-1 && any(k == [1 2 3])) || ...
        (i == m-2 && any(k == [2 3])) || ...
        (i == m-3 &&     k == 3)
        warning('Bpp: k not right value for for B_{m-1,m-2,m-3}\n');
        return;
    end

    assert((i >= 0 && i <= m-1),'Bpp: invalid basis i');
    l = i*h;
    r = (i+4)*h;

    switch k
    case 0
        Bx =  (x-l)/(h^3);
    case 1
        Bx =  4/h^2 - 3*(x-l)/(h^3);
    case 2
        Bx = -8/h^2 + 3*(x-l)/(h^3);
    case 3
        Bx =  (r-x)/(h^3);
    otherwise
        warning('Bpp: k ~\in {0,1,2,3} for B_i\n');
    end
end



    
    Note we can still use Gaussian Quadrature to evaluate the integral for $f_{k,l}$ in both axial directions as follows
    \[
        \int\int f(x,y) dxdy
            = \int \sum_{i=1}^n w_i f(x_i, y) dy
            = \sum_{i=1}^n\sum_{j=1}^n w_i w_j f(x_i,y_i)
    \]
    where $x_i,w_i$ are sample points and corresponding weights.