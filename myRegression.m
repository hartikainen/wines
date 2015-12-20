function [ testPreds ] = myRegression(trainX, trainY, testX)
%MYREGRESSION Simple linear regression

[Ntrain, dataDim] = size(trainX);

% we are substituting the different degrees with different dimensions of
% linear regressors
basisFunsTrain = ones(Ntrain, dataDim+1);
basisFunsTrain(:, 2:end) = trainX;

betasTmp = basisFunsTrain \ trainY;

% test data
Ntest = size(testX, 1);

basisFunsTest = ones(Ntest, dataDim+1);
basisFunsTest(:, 2:end) = testX;

testPreds = round(basisFunsTest * betasTmp);
end