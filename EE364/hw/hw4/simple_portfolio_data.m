%% simple_portfolio_data
rand('state', 5);
randn('state', 5);
n=20;
pbar = ones(n,1)*.03+[rand(n-1,1); 0]*.12;
S = randn(n,n);
S = S'*S;
S = S/max(abs(diag(S)))*.2;
S(:,n) = zeros(n,1);
S(n,:) = zeros(n,1)';
x_unif = ones(n,1)/n;


% (a)
cvx_begin quiet
    variables x(n);

    % minimize risk by return variance
    minimize quad_form(x,S);
        % same expected return as uniform portfolio
        pbar'*x == pbar'*x_unif;
        % long only
        x >= 0;
        % total to 1
        sum(x) == 1;
cvx_end


disp('var(x_uinf)');
quad_form(x_unif,S)

disp('var(x*)');
quad_form(x,S)