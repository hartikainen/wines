t1 = readtable('training_data.csv');
t2 = readtable('test_data.csv');

trainData = table2array(t1(:, 1:12));
testData = table2array(t2(:, 2:12));

N_TREES = 400;

perm = randperm(5000);
trainPerm = perm(1:4000);
testPerm = perm(4001:5000);

B = TreeBagger(N_TREES, trainData(trainPerm, 1:11), trainData(trainPerm, 12));

preds = str2num(cell2mat(B.predict(trainData(testPerm, 1:11))));

testErrs = preds - trainData(testPerm,12);
err = mean(testErrs.^2) 

% use these for test data

%B = TreeBagger(N_TREES, trainData(:, 1:11), trainData(:, 12));
%preds = str2num(cell2mat(B.predict(testData(:, 1:11))));
%f = fopen('output','w');
%fprintf(f,'id,quality\n');
%outputM = [table2array(t2(:,1)) preds]';
%fprintf(f,'%d,%d\n',outputM);
