function params = initialize_params_randn(inputSize,hiddenSize)
    params.layer1.lin_r_weight = dlarray(randn(inputSize,hiddenSize));
    params.layer1.lin_l_weight = dlarray(randn(inputSize,hiddenSize));
    params.layer1.lin_l_bias = dlarray(zeros(1,hiddenSize));

    params.layer2.lin_r_weight = dlarray(randn(hiddenSize, hiddenSize));
    params.layer2.lin_l_weight = dlarray(randn(hiddenSize, hiddenSize));
    params.layer2.lin_l_bias = dlarray(zeros(1,hiddenSize));

    params.layer3.lin_r_weight = dlarray(randn(hiddenSize, hiddenSize));
    params.layer3.lin_l_weight = dlarray(randn(hiddenSize, hiddenSize));
    params.layer3.lin_l_bias = dlarray(zeros(1,hiddenSize));

    params.classifier.weights = dlarray(randn(1, hiddenSize));
    params.classifier.bias = dlarray(zeros(1, 1));
end