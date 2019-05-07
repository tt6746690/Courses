function [coefs,values] = polyfitLS(patch, n)
%
% fit a polynomial of degree n to a patch of pixels
% [coefs eval] = function polyfitLS(patch, n)
%    patch must be a column vector of length > n+1
%

% get patch radius (assume an odd number of pixels)
w = floor(length(patch)/2);
% size of patch
N = 2*w+1;
x = (-w:w)';

% build coefficient & constant matrix
X = zeros(N,n+1);
for i=1:(n+1)
    X(:,i) = x.^(i-1);
end

coefs = pinv(X)*patch;
values = X*coefs;