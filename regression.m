poly_deg = 1;

clear all; close all;

% assume that N_train is divisible by FOLDK
FOLDK = 10;

training_data_table = readtable('training_data.csv');
test_data_table     = readtable('test_data.csv');

training_data = table2array(training_data_table(:,1:12));
test_data = table2array(test_data_table(:,1:12));

N_train = size(training_data, 1);

dim_data = 11;

perm = randperm(N_train);

% we are substituting the different degrees with different dimensions of
% linear regressors
basis_funs = ones(N_train, (11+1));
for i = 1:dim_data
    basis_funs(:, (i+1) ) = training_data(:,i);
end

valid_preds = zeros(N_train,1);

for i = 1:size(training_data,1)
    % learn model in training set
    training_idx = perm([1:i-1 i+1:end]);
    test_idx = perm(i);

    X_tmp = basis_funs(training_idx, 1:11);
    betas_tmp = pinv(X_tmp) * training_data(training_idx,12);
    valid_preds(test_idx) = basis_funs(test_idx, 1:11) * betas_tmp;
    train_preds = basis_funs(:, 1:11) * betas_tmp;
end

% compute training errors
training_errs = train_preds - training_data(:,12);

% compute validation errors
valid_errs = valid_preds - training_data(:,12);

% test data
n_test_data = size(test_data,1);

basis_funs_test_data = ones(n_test_data, (11 + 1) );

for i = 1:dim_data
    % test_data has id as the first column
    basis_funs_test_data(:, (i+1) ) = test_data(:,i+1);
end

test_preds =  round(basis_funs_test_data(:, 1:11) * betas_tmp);

f = fopen('output','w');
fprintf(f,'id,quality\n');
outputM = [test_data(:,1) test_preds]';
fprintf(f,'%d,%d\n',outputM);
