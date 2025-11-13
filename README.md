# LapSDNMF
Matlab implementation for "Label Propagation Assisted Soft-constrained Deep Non-negative Matrix Factorisation for Semi-supervised Multi-view Clustering"

If you find it useful, please consider citing our work.

```bibtex
@article{gunawardena2025lapsdnmf,
  title={LapSDNMF: Label propagation assisted soft-constrained deep non-negative matrix factorisation for semi-supervised multi-view clustering},
  author={Gunawardena, Sohan Dinusha Liyana and Luong, Khanh and Balasubramaniam, Thirunavukarasu and Nayak, Richi},
  journal={Machine Learning with Applications},
  pages={100783},
  year={2025},
  publisher={Elsevier},
  doi={https://doi.org/10.1016/j.mlwa.2025.100783}
}
```

Abstract: Semi-supervised methods based on non-negative matrix factorisation have emerged as a popular approach for clustering. 
However, the pressing challenge of capturing complex non-linear relationships within multi-view data is seldom considered in the semisupervised context. 

This study introduces a fundamentally novel framework: Label Propagation Assisted Soft-constrained Deep Non-negative Matrix Factorisation for Semi-supervised Multi-view Clustering (LapSDNMF). 

LapSDNMF innovatively integrates deep hierarchical modelling with label propagation and soft constraint to jointly exploit the non-linear representation learning and extract accurate latent features from limited labelled data. 
By embedding a predictive membership matrix as a soft constraint, it enables similarly labelled samples to be projected into shared regions, better reflecting real-world data structures. 
The incorporation of graph-based regularisation within the deep architecture facilitates effective label propagation while preserving the manifold structure at each layer. 
LapSDNMF unifies deep learning and graph-theoretic techniques within a coherent optimisation framework. 
We also develop a novel, efficient algorithm based on multiplicative update rules to solve the resulting optimisation problem. 

LapSDNMF significantly outperforms state-of-the-art multi-view clustering methods across five diverse real-world datasets.
Specifically, it achieves improvements in F-score of 10.2%, 7.2%, 8.8%, 1.4%, and 6.1% on the Yale, Reuters-MinMax, Caltech7, 3-Sources, and Caltech20 datasets, respectively, compared with the best-performing baseline method.


# Demo
Run "Main.m" to see the provided example of Yale dataset.
