
A = [-1 0.4 0.8; 1 0 0; 0 1 0];
b = [1;0;0.3];
xdes = [7;2;-6];

n = 3; % state dimension
N = 30;

H = zeros(size(A,1),N);
for i = 1:N
    H(:,i) = A^(N-i)*b;
end


% The equivalent linear program 
% 
cvx_begin quiet
    variables u(N) l(N);

    minimize sum(l);
        H*u == xdes;
        -l <= u <= l;
        2*u-1 <= l;
        -2*u-1 <= l;
cvx_end

% Or use CVX directy
%
% x0 = zeros(n,1);
% cvx_begin
%     variable X(n,N+1);
%     variable u(1,N);

%     minimize sum(max(abs(u), 2*abs(u)-1));
%     subject to 
%         X(:,2:N+1) == A*X(:,1:N) + b*u;
%         X(:,1) == x0;
%         X(:,N+1) == xdes;
% cvx_end

stairs(0:N-1,u,'linewidth',2);
axis tight;
xlabel('t');
ylabel('u');