function [loss, grad] = modelLoss(parameters,features,edges,targets,numNodes)

out = model(parameters,features,edges,numNodes); %Model defined here
loss = crossentropy(out, targets', DataFormat='CB', ClassificationMode='multilabel');

% Gradient
grad = dlgradient(loss, parameters);
end