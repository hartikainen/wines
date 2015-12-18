function [ standardized ] = standardize( X )
%STANDARDIZE Standardizes the given input matrix X
%   Returns zero mean, unit variance matrix of X

    X = double(X);
    meaned = X - repmat(mean(X), size(X, 1), 1);
    standardized = meaned ./ repmat(std(meaned), size(meaned, 1), 1);
end

