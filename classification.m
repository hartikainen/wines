clear all; close all;

MAX_K = 3; % maximum k neighbors
STANDARDIZE = 1;
MAX_FOLDS = 10;

trainData = readtable('training_data.csv');
testData     = readtable('test_data.csv');

test = MyKnnClass(trainData{:, 1:12}, trainData{:, 13}, ...
                  STANDARDIZE, MAX_FOLDS);

test.train(MAX_K);