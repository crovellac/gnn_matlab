tf = canUseGPU;
if tf
    disp('GPU available.')
else
    disp('GPU not available.')
end

%% Load training data
load('train_graphs.mat');

features_table = train_graphs.Features;
edges_table = train_graphs.Adjacency;
targets_table = train_graphs.y;

%% Parameters
inputSize = 1;
numClasses = 2;
hiddenSize = 64;

parameters = initialize_params_glorot(inputSize,hiddenSize);
%parameters = initialize_params_randn(inputSize,hiddenSize);

%% Train Hyperparameters
numEpochs = 100;
learnRate = 0.0001;
velocity = [];
momentum = 0.9; %for sgdm
trailingAvg = [];  %for adam
trailingAvgSq = [];  %for adam
iteration=1;  %for adam

numTrain = 29000
batchSize = 100
numBatches = numTrain/batchSize

epoch = 0;
% Loop over epochs.
while epoch < numEpochs
    shuffle_idx = randperm(height(train_graphs));
    epoch = epoch + 1;
    for batch=1:numBatches
        features = [];
        numNodesPerGraph = [];
        edges = [];
        targets = [];
        for i=1:batchSize
            index = shuffle_idx(i+(batch-1)*batchSize);
            features = [features; cell2mat(features_table(index))];
            numNodesPerGraph = [numNodesPerGraph (length(cell2mat(features_table(index))))];
            edges = blkdiag(edges,cell2mat(edges_table(index)));
            %targets = [targets; [targets_table(index) abs(1-targets_table(index))]];
            targets = [targets; targets_table(index)];
        end
        features = dlarray(features);
        numNodes = sum(numNodesPerGraph);
        edges = dlarray(boolean(edges));
        targets = double(targets);
        if tf
            features = gpuArray(features);
            edges = gpuArray(edges);
        end
        % Evaluate the model loss and gradients using dlfeval and the
        % modelLoss function
        [loss,gradients] = dlfeval(@modelLoss,parameters,features,edges,targets,numNodesPerGraph);
        % Update the network parameters using the SGDM optimizer.
        %[parameters,velocity] = sgdmupdate(parameters,gradients,velocity,learnRate,momentum);
	[parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients,trailingAvg,trailingAvgSq,iteration,learnRate);
	iteration = iteration+1;
    end
    %Display loss at end of epoch
    fprintf("Epoch: %d, Loss: %f \n", epoch, loss);
    if mod(epoch,10) == 0
	fname = strcat('models/modelparams_epoch',string(epoch),'.mat');
	save(fname,'parameters');
    end
end

%% Save model
save('modelparams.mat','parameters');

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
