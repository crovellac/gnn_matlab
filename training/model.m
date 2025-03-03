%Model function
function out = model(parameters,features,edges,numNodes)
    %GraphSAGE Layer 1
    x1 = edges*features*parameters.layer1.lin_l_weight;
    x1 = x1 + features*parameters.layer1.lin_r_weight;
    x1 = x1 + parameters.layer1.lin_l_bias;
    %ReLu and Normalize
    x1 = relu(x1);
    x1 = normalize(x1, 2, 'norm');
    %Pooling 1
    x1_readout = graphAveragePool(x1,numNodes);

    %GraphSAGE Layer 2
    x2 = edges*x1*parameters.layer2.lin_l_weight;
    x2 = x2 + x1*parameters.layer2.lin_r_weight;
    x2 = x2 + parameters.layer2.lin_l_bias;
    %ReLu and Normalize
    x2 = relu(x2);
    x2 = normalize(x2, 2, 'norm');
    %Pooling 1
    x2_readout = graphAveragePool(x2,numNodes);

    x_readout = x1_readout+x2_readout;
    
    %Classification
    out = parameters.classifier.weights*x_readout';
    out = out+parameters.classifier.bias;
    out = sigmoid(out);
end