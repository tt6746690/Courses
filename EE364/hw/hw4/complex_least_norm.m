% complex minimum norm problem 

randn('state',0);
m = 30; n = 100;

RA = randn(m,n); CA = randn(m,n);
Rb = randn(m,1); Cb = randn(m,1);

A = complex(RA,CA);
b = complex(Rb,Cb);

% 2-norm reformulated to least squared
AA = [RA -CA; CA RA];
bb = [Rb;Cb];
z_2 = AA'*inv(AA*AA')*bb;
x_2 = complex(z_2(1:100,1),z_2(101:200,1));

% 2-norm with cvx
cvx_begin quiet
    variable x(n) complex;
    minimize norm(x,2);
        A*x == b;
cvx_end

% approx same
(x(1:5)-x_2(1:5))'

% inf-norm with cvx
cvx_begin quiet
    variable xinf(n) complex;
    minimize norm(xinf,Inf);
        A*xinf == b;
cvx_end

cvx_status
xinf(1:5)'


% scatter 
figure(1);
scatter(real(x),imag(x)), hold on,
scatter(real(xinf),imag(xinf),[],'filled'), hold off,
axis([-0.2 0.2 -0.2 0.2]), axis square,
xlabel('Re x'); ylabel('Im x');
