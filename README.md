# Robust Graph Neural Networks – Modified NRGNN for noisy edge 
Motivated by the [NRGNN](https://github.com/EnyanDai/NRGNN)[1], this project aims to consider the model robustness with modified NRGNN model.   
- According to the NRGNN model, the following questions are discussed in this project:
1. Pseudo Label Miner and GNN classifier share similar function. Can we just update one model?
2. Pseudo edges link unlabeled node and labeled node. Higher similarities are expected for these node pairs.
3. Noisy edges: NRGNN do not modify the weights of labeled edges Weights of labeled edges are 1 throughout training process. Can we progressively reweight the edge weights as if they were pseudo labels?

- To solve these problems, this work modified the original NRGNN model:
1. Introduced the concept of Mean Teacher[2] to imrpove the prediction consistency of the pseudo label miner and the GNN classifier.

![image](https://user-images.githubusercontent.com/42937407/181176225-acf97786-5018-41e0-97ed-69f94843fcf2.png)

2. Inspired by the [3],pseudo edges link unlabeled node and labeled node and igher similarities are expected for these node pairs. The method of label smoothness on unlabeled nodes is also intoduced. 

3. To mitigate the negative effect of noisy labels, this work proposed reweight labeled edges to reassign the wight of edges according to the feature similarity between the node pair. 

![image](https://user-images.githubusercontent.com/42937407/181177109-8868da0e-5df8-4780-99d6-f8edf3b55ad7.png)

## Experiment results
Table 1. Accuracy of GNN classifier on cora dataset under noisy labels and noisy edges

![image](https://user-images.githubusercontent.com/42937407/181178069-29a020c3-5a81-4106-b729-d6567271f2a3.png)

Table 2. Accuracy of GNN classifier on citeseer dataset under noisy labels and noisy edges

![image](https://user-images.githubusercontent.com/42937407/181178099-8a405df6-2472-49db-83df-621672e7cd8c.png)

---
This work and the original NRGNN framework assume the given edges connect high similarity features. To varift this, we check the node pair feature
similarity of all given edges.

(a) Cora dataset

![image](https://user-images.githubusercontent.com/42937407/181177380-ae92b98a-3526-49ba-b702-8a38b8b9737d.png)

(b) Citeseer dataset

![image](https://user-images.githubusercontent.com/42937407/181177470-57fd6664-79eb-444d-bffa-ce0fe843f262.png)

The experiment shows the node features are quite different and less than 10% given edges have high confidence in terms of node pair similarities. In NRGNN,
this observation may harm the assumption and cause poor performance. Further investigation should be done in the futureworks.


The experiments show NRGNN could not perform well on noisy edges. Mean teacher frameworks do not significantly improve the model. Besides, that may
harm the performance. Reweighting the edges gives certain resistance for the edge noise. To resist more attack, attributes noise should be considered for future works. Finally, this work also shows the node pair similarity for existing node pairs. The results should be considered for the following works.

---
[1] E. Dai, C. Aggarwal, and S. Wang, “NRGNN: Learning a Label Noise-Resistant Graph Neural Network on Sparsely and Noisily Labeled Graphs,” 2021.

[2] A. Tarvainen and H. Valpola, “Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,” 2018.

[3] E. Dai, W. jIN, H. Liu, and S. Wang, “Towards Robust Graph Neural Networks for Noisy Graphs with Sparse Labels.” Jan. 01, 2022
