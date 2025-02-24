%% Model function
function [loss, grad] = modelLoss(parameters,features,edges,targets,numNodes)

% GraphSAGE model

%x1
x1 = graphSageMean(features, edges, parameters.layer1);
x1 = relu(x1);
x1 = normalize(x1, 2, 'norm');
% Graph pooling for graph classification
x1_readout = graphAveragePool(x1, numNodes);

%x2
x2 = graphSageMean(x1, edges, parameters.layer2);
x2 = relu(x2);
x2 = normalize(x2, 2, 'norm');
% Graph pooling for graph classification
x2_readout = graphAveragePool(x2, numNodes);

%x3
x3 = graphSageMean(x2, edges, parameters.layer3);
x3 = relu(x3);
x3 = normalize(x3, 2, 'norm');
% Graph pooling for graph classification
x3_readout = graphAveragePool(x3, numNodes);

out = x1_readout+x2_readout+x3_readout;

% Classifier
weights = parameters.classifier.weights;
bias = parameters.classifier.bias;

out = fullyconnect(out, weights, bias, DataFormat='BC');
%out = softmax(out, DataFormat="BC");
out = sigmoid(out);

%out
%targets'

% Loss
loss = crossentropy(out, targets', DataFormat='CB', ClassificationMode='multilabel');

% Gradient
grad = dlgradient(loss, parameters);
end
