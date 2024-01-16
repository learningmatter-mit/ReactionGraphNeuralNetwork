# ReactionGraphNeuralNetwork
Graph Neural Network to predict the reaction related properties for reinforcement learning.

## Installation

- install pytroch, torch_geometric, torch_scatter, torch_cluster pytorch-sparse with right cuda version below example is for pytorch version 2.1 with cuda version 12.1

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg pytorch-cluster pytorch-scatter pytorch-sparse -c pyg 
```