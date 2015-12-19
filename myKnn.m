function [ totalAcc, predictions ] = myKnn(trainX, trainY, testX, k, folds)
%MYKNN Simple k-nearest neighbor classifier
%   Cross-validate the classification accuracy using the training data,
%   i.e. trainX and trainY. Predict the classes for testX.

trainN = size(trainX, 1);
X = standardize(trainX);
Y = zeros(trainN, 1);
classes = unique(trainY);

for i=1:length(classes)
    Y(strcmp(trainY, classes(i))) = i;
end

% Takes as an input N1 x D matrix X1 and N2 x D matrix X2.
% returns a N2 x K matrix, including k nearest neighbours in X
% for each point in Y.
% Find the nearest neighbors f

idx = crossvalind('Kfold',trainN,folds);

totalAcc = 0;
for fold=1:folds
    validIdx = (idx == fold);
    trainIdx = ~validIdx;

    X1 = X(trainIdx, :);
    Y1 = Y(trainIdx, :);
    X2 = X(validIdx, :);
    Y2 = Y(validIdx, :);
    
    % find the neighbors for the validation points
    distances = pdist2(X1, X2);
    [d, I] = sort(distances);
    neighbors = I(1:k, :)';
    
    % for each validation points, find the most frequent classes of the neighbors
    predicted = mode(Y1(neighbors), 2);
    acc = sum(predicted == Y2) / length(predicted);
    totalAcc = totalAcc + acc;
end
totalAcc = totalAcc / folds;

% find the neighbors for the validation points
distances = pdist2(trainX, testX);
[d, I] = sort(distances);
neighbors = I(1:k, :)';

testKey = mode(Y(neighbors), 2);
predictions = classes(testKey);

end

