function [coefs,values,weight] = polyfitWLS(patch, n)
%
% fit a polynomial of degree n to a patch of pixels
% using weighted least squares
%
%  [coefs eval omega] = function polyfitWLS(patch, n)
%     patch must be a column vector of length > n+1
%

% get patch radius (assume an odd number of pixels)
w = floor(length(patch)/2);
% size of patch
N = 2*w+1;
x = (-w:w)';

% use sigma = radius/2
sigma = w/2;
weight = (1/(sigma*sqrt(2*pi)))*exp(-x.^2/sigma^2);
weight = weight/sum(weight);
omega = diag(weight);

% build coefficient & constant matrix
X = zeros(N,n+1);
for i=1:(n+1)
    X(:,i) = x.^(i-1);
end

coefs = pinv(omega*X)*(omega*patch);
values = X*coefs;

weight = 256*weight/max(weight);