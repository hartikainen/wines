poly_deg = 1;

clear all; close all;

% assume that N_train is divisible by FOLDK
FOLDK = 10;

training_data_table = readtable('training_data.csv');
test_data_table     = readtable('test_data.csv');

training_data = table2array(training_data_table(:,1:12));
test_data = table2array(test_data_table(:,1:12));

N_train = size(training_data, 1);
data_dim = size(training_data, 2) - 1;

perm = randperm(N_train);

% we are substituting the different degrees with different dimensions of
% linear regressors
basis_funs = ones(N_train, (data_dim+1));
basis_funs(:, 2:end) = training_data(:, 1:data_dim);

valid_preds = zeros(N_train,1);

betas_tmp = basis_funs \ training_data(:, 12);

% test data
n_test_data = size(test_data,1);

basis_funs_test = ones(n_test_data, data_dim + 1);
basis_funs_test(:, 2:end) = test_data(:, 2:data_dim+1);

test_preds =  round(basis_funs_test * betas_tmp);

f = fopen('output','w');
fprintf(f,'id,quality\n');
outputM = [test_data(:,1) test_preds]';
fprintf(f,'%d,%d\n',outputM);
