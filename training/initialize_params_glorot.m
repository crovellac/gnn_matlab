function params = initialize_params_glorot(inputSize,hiddenSize)
    params.layer1.lin_r_weight = dlarray(initializeGlorot([inputSize hiddenSize],inputSize,hiddenSize));
    params.layer1.lin_l_weight = dlarray(initializeGlorot([inputSize hiddenSize],inputSize,hiddenSize));
    params.layer1.lin_l_bias   = dlarray(zeros(1,hiddenSize));

    params.layer2.lin_r_weight = dlarray(initializeGlorot([hiddenSize hiddenSize],hiddenSize,hiddenSize));
    params.layer2.lin_l_weight = dlarray(initializeGlorot([hiddenSize hiddenSize],hiddenSize,hiddenSize));
    params.layer2.lin_l_bias   = dlarray(zeros(1,hiddenSize));

    params.layer3.lin_r_weight = dlarray(initializeGlorot([hiddenSize hiddenSize],hiddenSize,hiddenSize));
    params.layer3.lin_l_weight = dlarray(initializeGlorot([hiddenSize hiddenSize],hiddenSize,hiddenSize));
    params.layer3.lin_l_bias   = dlarray(zeros(1,hiddenSize));

    params.classifier.weights = dlarray(initializeGlorot([1, hiddenSize],1,hiddenSize));
    params.classifier.bias = dlarray(zeros(1, 1));
end