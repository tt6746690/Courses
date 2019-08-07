
% corresponds to 4.17 

A = [
    1 2 0 1
    0 0 3 1
    0 3 1 1
    2 1 2 5
    1 0 3 2
];

c = [100;100;100;100;100];
p = [3;2;7;6];
pdisc = [2;1;4;2];
q = [4;10;5;10];


cvx_begin quiet
    variable x(4);
    maximize sum(min(p.*x, p.*q + pdisc.*(x-q)));
        A*x <= c;
        x >= 0;
cvx_end


if isequal(cvx_status,'Solved')
    disp('optimal activity level: ');
    disp(x);
    disp('revenue: ');
    disp(min(p.*x, p.*q + pdisc.*(x-q)));
    disp('total revenue: ');
    disp(cvx_optval);
    disp('average price: ');
    disp(r./x);
else
    disp(cvx_status);
    disp(cvx_optval);
end
