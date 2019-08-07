% 4.15; 5.13

rand('state',0);
n=100;
m=300;
A=rand(m,n);
b=A*ones(n,1)/2;
c=-rand(n,1);

% solve the relaxed problem 

cvx_begin quiet
    variable x(n);
    minimize dot(c,x);
        A*x <= b;
        0 <= x <= 1;
cvx_end

L = cvx_optval; % -33.17

% round and compute objective value and maximum constraint violation

ts = linspace(0,1,100);
obj = zeros(size(ts));
maxviol = zeros(size(ts));

for i = 1:size(ts,2)
    t = ts(i);
    xhat = (x >= t);
    obj(i) = dot(c,xhat);
    maxviol(i) = max(A*xhat-b);
end


% find t for which xhat is feasible, and gives minimum objective value
%   note upper bound U=c^T*xhat, compute the gap U-L
i_feas = find(maxviol<=0);
U = min(obj(i_feas));           % -32.4450
ti = min(i_feas);
t = ts(ti);                     % 0.6061
fprintf('U-L=%.3f\n',U-L);      % 0.7222

% boolean relaxation on LP works surprisingly well in this setup...
%       real world problem can be quite different...


% obj/max violation w.r.t. threshold plot

figure;
subplot(2,1,1);
plot(ts(ti:end),maxviol(ti:end),'r',ts(1:ti-1),maxviol(1:ti-1),'g');
xlabel('threshold');
ylabel('maximum violation');
title('maximum constraint violation (green: feasible)');

subplot(2,1,2);
plot(ts(ti:end),obj(ti:end),'r',ts(1:ti-1),obj(1:ti-1),'g'); hold on;
plot(ts,L*ones(size(maxviol)));
xlabel('threshold');
ylabel('objective value');
title('objective value for rounded xhat (green:feasible)');

