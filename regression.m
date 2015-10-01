clear all; close all;

% assume that N_train is divisible by FOLDK
FOLDK = 5;

training_data = readtable('training_data.csv');
test_data     = readtable('test_data.csv');

N_train = size(training_data, 1);
N_test = size(test_data, 1);

I = randperm(N_train);

for k=1::N_train

end