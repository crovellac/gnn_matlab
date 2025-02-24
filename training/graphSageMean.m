function out = graphSageMean(features, edges, parameters, combinationFun)
%graphSageMean   GraphSAGE model with mean aggregator
%
% Inputs:
% features:                  Node features - N x F dlarray
% edges:                     Edge information - N x N adjacency matrix
% parameters:                Learnable parameters - 1x1 struct with fields
%                            "layerK.weights.self" and
%                            "layerK.weights.neighbor", where K = 1,2,...
%                            represents layer index.
% numSamples (optional):     Number of nodes to sample in each layer - 1x1
%                            struct with field names "layerK", where K =
%                            1,2,... represents layer index.
% combinationFun (optional): Function for combining self and neighbor
%                            features, specified as "cat" (default) or
%                            "add".
%
% Output:
% out:                       N x H dlarray. If combinationFun is 'cat', H
%                            is 2*output size of the last layer, else, H is
%                            output size of the last layer.

arguments
    features {mustBeA(features, 'dlarray')}
    edges
    parameters {mustBeA(parameters,'struct')}
    combinationFun {mustBeMember(combinationFun, ["cat", "add"])} = 'add'
end

dimLabels = dims(features);
features = stripdims(features);
combinationFun = iCombinationFun(combinationFun);

% Layer 1
selfWeights = parameters.lin_r_weight;
neighborWeights = parameters.lin_l_weight;

features = graphSage(features);

if isempty(dimLabels)
    out = features;
else
    out = dlarray(features, dimLabels);
end

    function nodeFeatures = graphSage(nodeFeatures)
        % Sample and aggregate neigbour features
        neighborFeatures = aggregate(nodeFeatures);

        % Project
        nodeFeatures = mtimes(nodeFeatures, selfWeights);
        neighborFeatures = mtimes(neighborFeatures, neighborWeights);
        
        % Combine
        nodeFeatures = combinationFun(nodeFeatures, neighborFeatures);
    end

    function neighborFeatures = aggregate(nodeFeatures)
        [numNodes, numFeatures] = size(nodeFeatures);
        neighborFeatures = zeros(numNodes, numFeatures);
        
        %TODO: Replace with vectorized version
        for i = 1:numNodes
            neighbors = find(edges(i, :));
            neighborFeatures(i,:) = mean(nodeFeatures(neighbors',:));
        end
    end

end

function fun = iCombinationFun(fun)
switch fun
    case 'cat'
        fun = @(x,y)cat(2,x,y);
    case 'add'
        fun = @(x,y)x+y;
end
end
