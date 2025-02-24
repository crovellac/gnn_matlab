# gnn_matlab
A graph neural network implementation in MATLAB for physics event classification, targeting FPGA deployment for use as a hardware trigger.

## Contents

- `data`: Contains the graph data used for training/testing.
- `graph_generation`: Converts the original [Zenodo Top Quark Tagging Dataset](https://zenodo.org/records/2603256#.YIgG8R-SlaR) into graphs, which are saved in `data`.
- `models`: Contains saved model parameters after training. These parameters can then be loaded to test the model.
- `training`: The scripts for training the GraphSAGE model.
- `deploy`: Project for quantizing the model and deploying it on an FPGA.
