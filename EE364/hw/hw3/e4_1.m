

A = -[2 1; 1 3];
b = -[1;1];

fprintf(1,'(a)\n');
cvx_begin quiet
    variables x(2);
    minimize sum(x);
        A*x <= b;
        x >= 0;
cvx_end
if isequal(cvx_status,'Solved')
    disp('x: ');
    disp(x);
    disp('optimal value:');
    disp(cvx_optval);
else
    disp(cvx_status);
end


fprintf(1,'(b)\n');
cvx_begin quiet
    variables x(2);
    minimize -sum(x);
        A*x <= b;
        x >= 0;
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



fprintf(1,'(c)\n');
cvx_begin quiet
    variables x(2);
    minimize x(1);
        A*x <= b;
        x >= 0;
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


fprintf(1,'(d)\n');
cvx_begin quiet
    variables x(2);
    minimize max(x);
        A*x <= b;
        x >= 0;
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



fprintf(1,'(e)\n');
cvx_begin quiet
    variables x(2);
    minimize x(1)^2 + 9*x(2)^2
        A*x <= b;
        x >= 0;
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
