clear all; % close all;

OUTPUT = 'output/qualities.csv';

trainData = readtable('training_data.csv');
testData  = readtable('test_data.csv');
voittoCsv = readtable('voitto1.csv');

% TODO: table assignment aint working
testData(:, 13) = voittoCsv(:, 2); % assign the predicted types
testId = testData{:, 1};
testData = testData(:, 2:end);

trainRedIdx = strcmp(trainData{:, 13}, 'Red');
trainRedTable   = trainData(trainRedIdx, :);
trainWhiteTable = trainData(~trainRedIdx, :);

trainWhiteX = standardize(trainWhiteTable{:, 1:11});
trainWhiteY = trainWhiteTable.quality;
trainRedX    = standardize(trainRedTable{:, 1:11});
trainRedY    = trainRedTable.quality;

testRedIdx = strcmp(voittoCsv{:, 2}, 'Red');
testRedTable   = testData(testRedIdx, :);
testWhiteTable = testData(~testRedIdx, :);

testWhiteX = standardize(testWhiteTable{:, 1:11});
testWhiteY = testWhiteTable.quality;
testRedX    = standardize(testRedTable{:, 1:11});
testRedY    = testRedTable.quality;

whiteQualities = whitequality(trainWhiteX, ...
                              trainWhiteY, ...
                              testWhiteX);

redQualities = redquality(trainRedX, ...
                          trainRedY, ...
                          testRedX);

qualities(testRedIdx)  = redQualities;
qualities(~testRedIdx) = whiteQualities;

% f = fopen(OUTPUT,'w');
% fprintf(f,'id,quality\n');
% fprintf(f,'%d,%d\n', [testId qualities']');
