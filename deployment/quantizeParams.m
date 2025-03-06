function qparams = quantizeParams(parameters)
    p = fi([],1,16,10);
    qparams.layer1.lin_r_weight= cast(parameters.layer1.lin_r_weight,'like',p);
    qparams.layer1.lin_l_weight = cast(parameters.layer1.lin_l_weight,'like',p);
    qparams.layer1.lin_l_bias = cast(parameters.layer1.lin_l_bias,'like',p);

    qparams.layer2.lin_r_weight = cast(parameters.layer2.lin_r_weight,'like',p);
    qparams.layer2.lin_l_weight = cast(parameters.layer2.lin_l_weight,'like',p);
    qparams.layer2.lin_l_bias = cast(parameters.layer2.lin_l_bias,'like',p);

    qparams.layer3.lin_r_weight = cast(parameters.layer3.lin_r_weight,'like',p);
    qparams.layer3.lin_l_weight = cast(parameters.layer3.lin_l_weight,'like',p);
    qparams.layer3.lin_l_bias = cast(parameters.layer3.lin_l_bias,'like',p);

    qparams.classifier.weights = cast(parameters.classifier.weights,'like',p);
    qparams.classifier.bias = cast(parameters.classifier.bias,'like',p);
end