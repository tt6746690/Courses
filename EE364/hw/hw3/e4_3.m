
P = [13 12 -2; 12 17 6; -2 6 12];
q = [-22;-14.5;13];
r = 1;

cvx_begin quiet
    variable x(3);
    % minimize ((1/2)*x'*P*x + q'*x + r);  % or
    minimize ((1/2)*quad_form(x,P) + q'*x + r);
         -1<=x<=1;
cvx_end


if isequal(cvx_status,'Solved')
    disp('x: ');
    disp(x);
    disp('optimal value:');
    disp(cvx_optval);
else
    disp(cvx_status);
    disp(cvx_optval);
end
