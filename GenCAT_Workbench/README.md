# README

## GenCAT Workbench

In submission.

## Python notebooks
For the reproduction of our demo, we provide a Jupyter notebook, ```GenCAT_Workbench_Usage_and_UseCase.ipynb```.

This notebook consists of two parts:
+ Usage of parameter configuration and graph generation
+ Use cases for demonstrating deep analysis on graph neural networks with various synthetic graphs
  + Accuracy on graphs with various "edge connection proportions between classes".
  + Accuracy on graphs with various "attribute values".
  + Training time per epoch for various "number of edges".

Also, the notebooks in a folder "detailed_notebooks" show other detailed experiments.

## Source code of GenCAT

+ gencat.py
  + We utilize the source code of GenCAT.
  + https://arxiv.org/abs/2109.04639

### Requirements
For using GenCAT
+ jgraph==0.2.1
+ powerlaw==1.4.6

Other packages are pre-installed in Colab.

## Source codes of GNNs
  + gcn-master (https://github.com/tkipf/gcn)
  + GAT-master (https://github.com/PetarV-/GAT)
  + H2GCN-master (https://github.com/GemsLab/H2GCN)
  
### Acknowledgements
We would like to thank the community for releasing their codes! This repository contains the codes from GCN, GAT, and H2GCN repositories.
We made small changes in their codes to output results.
