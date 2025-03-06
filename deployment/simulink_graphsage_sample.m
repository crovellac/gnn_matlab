%Load the data for a graph
load("feats_example.mat");
load("adj_example.mat");

maxNumNodes = 35; 
hiddenSize = 64; 

sz=height(feats);
feats=padarray(feats,maxNumNodes-sz,0,"post");
adj=padarray(adj,[maxNumNodes-sz maxNumNodes-sz],"post");

load("../models/modelparams_epoch80.mat","parameters");

parameters = quantizeParams(parameters);

parameters.layer1.lin_l_bias = [repmat(parameters.layer1.lin_l_bias,sz,1) ; zeros(maxNumNodes-sz,hiddenSize)];
parameters.layer2.lin_l_bias = [repmat(parameters.layer2.lin_l_bias,sz,1) ; zeros(maxNumNodes-sz,hiddenSize)];
parameters.layer3.lin_l_bias = [repmat(parameters.layer3.lin_l_bias,sz,1) ; zeros(maxNumNodes-sz,hiddenSize)];

open_system("graphsage_withdelay")