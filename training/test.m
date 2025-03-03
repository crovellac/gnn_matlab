tf = canUseGPU;
if tf
    disp('GPU available.')
else
    disp('GPU not available.')
end

%% Load model
load('models/modelparams_epoch80.mat');

%% Test
load('test_graphs.mat');
numTest = 1000;
trues = zeros([1 numTest]);
preds = zeros([1 numTest]);

new_shuffle_idx = randperm(height(test_graphs));
for i=1:numTest
    index = new_shuffle_idx(i);
    features = cell2mat(test_graphs.Features(index));
    features = dlarray(features);
    edges = cell2mat(test_graphs.Adjacency(index));
    edges = dlarray(boolean(edges));
    targets = test_graphs.y(index);
    targets = double(targets);
    trues(i) = targets;
    numNodes = length(features);
    preds(i) = model(parameters,features,edges,numNodes);
end

roundpreds = round(preds);
accuracy=nnz(trues==round(preds))/numTest
mtest = confusionmat(trues,roundpreds)
save('confusion_matrix_test.mat','mtest');

[X,Y,~,AUC] = perfcurve(trues,preds,1.0);
save('roc_curve_test.mat','X','Y');
AUC
