# Benchmarking GNNs with GenCAT Workbench

This paper is accepted by the demo track of [ECML/PKDD 2022](https://2022.ecmlpkdd.org/).

The successor paper is accepted to NeurIPS 2022 Datasets and Benchmarks Track [[paper](https://openreview.net/forum?id=bSULxOy3On)] [[code](https://github.com/seijimaekawa/empirical-study-of-GNNs)].


## Experimental reproducibility
For the reproduction of our demo, we provide a Jupyter notebook, ```GenCAT_Workbench_Usage_and_UseCase.ipynb```.

This notebook consists of two parts:
+ Usage of parameter configuration and graph generation
  + Parameter extraction from a given graph.
  + Parameter configuration for generating various graphs that users desire. 
  + Generating graphs and showing the statistics of the generated graphs.
+ Use cases for demonstrating deep analysis on graph neural networks with various synthetic graphs
  + Accuracy on graphs with various "edge connection proportions between classes".
  + Accuracy on graphs with various "attribute values".
  + Training time per epoch for various "number of edges".

Also, the notebooks in a folder "detailed_notebooks" show other detailed experiments.

### Implementation
We measure training time on a NVIDIA Tesla P100 GPU (12GB) and Intel(R) Xeon(R) CPU @ 2.20GHz (24GB).

## Demo video
We provide demo video of GenCAT Workbench, which explains the usage and use cases in detail.
+ [YouTube](https://www.youtube.com/watch?v=28xVOHRDpCE)
+ [MP4](https://drive.google.com/file/d/1Z8WarlXKFXHd-c1-J46AVwVnI98Djaen)

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
