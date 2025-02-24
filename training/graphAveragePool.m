function out = graphAveragePool(in, numNodes)
%graphAveragePool   Average graph pooling operation
%
% Inputs:
% in:       Features - N x F
% numNodes: Number of nodes per graph - 1 x G
%
% Output:
% out:      Pooled features - G x F

numGraphs = numel(numNodes);
numFeatures = size(in,2);
out = zeros(numGraphs, numFeatures, like=in);
startInd = 1;
endInd = 0;
for i = 1:numGraphs
    endInd = endInd + numNodes(i);
    out(i,:) = mean(in(startInd:endInd,:));
    startInd = endInd+1;
end
end