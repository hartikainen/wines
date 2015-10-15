OUTPUT_FILE = 'voitto.csv';

trainData = readtable('training_data.csv');
testData  = readtable('test_data.csv');
testId    = testData{:, 1};
testData  = testData(:, 2:end);

options = statset('MaxIter', 1000000);
svm = svmtrain(trainData{:, 1:11}, trainData{:, 13}, ...
               'options', options, 'kernel_function', 'rbf', ...
               'rbf_sigma', 3.75, 'boxconstraint', inf);

prediction = svmclassify(svm, testData{:, 1:11});

f = fopen('voitto.csv','w');
fprintf(f,'id,type\n');
for i=1:length(testId)
    fprintf(f,'%d,%s\n', testId(i), prediction{i,:});
end
