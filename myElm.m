function [ totalAcc, predictions ] = myElm( trainX, trainY, testX, ...
                                            lM, nlM )
%MYELM Simple extreme learning machine
%
%   lM - the number of linear neurons
%   nlM - the number of non-linear neurons

SIGMA = @tansig; % non-linear activation function

X = standardize(trainX)';
Y = trainY';
testX = standardize(testX)';

[n0, Ntrain] = size(X); % number of samples and dimensions

% create random weights
lW    = eye(n0+1);
lW(lM:end, :) = 0; % set the bias linear weight to 0
nlW = unifrnd(-3, 3, [nlM, n0+1]); % for uniform -3...3

biasX = [X; ones(1, Ntrain)];
H = [lW * biasX;
     SIGMA(nlW * biasX);
     ones(1, Ntrain)]';

pseudoinv = pinv(H);
beta = pseudoinv * Y';
P = H*pseudoinv;

e_loo = (Y' - P*Y') ./ (ones(length(Y'), 1) - diag(P));
e = (e_loo' * e_loo) / Ntrain;

Ntest = size(testX, 2);
biasTestX = [testX; ones(1, Ntest)];
Htest = [lW * biasTestX;
     SIGMA(nlW * biasTestX);
     ones(1, Ntest)]';
 
predictions = beta' * Htest';
totalAcc = 1-e;
end

