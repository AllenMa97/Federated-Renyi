# Federated-Renyi-Fair-Inference

Fairness in machine learning has been increasingly more important in recent years due to social responsibility.
Even if many algorithms have been proposed to address fairness concerns by introducing constraints or regularization in the conventional setting, only a few can deal with federated setting, where local private data is prohibited from sharing across users and central servers,
and estimation in the population level can be painfully difficult.
In this paper, we propose a `robust and efficient` algorithm that ensures R\'enyi-based fairness and can be well suited in the federated setting.
The proposed algorithm aggregates local statistics required by the R\'enyi correlation with a `simple yet robust` estimator, `under the challenge of federated privacy` constraints.
Our analysis provides decent concentration between such local aggregated empirical estimation and global population in $\tilde O(1/\sqrt{n})$ with high probability under mild conditions, where $n$ stands for the total number of data used in the federated setting.
Experiments on benchmark datasets verify the efficacy and robustness of our proposed algorithm.
Our algorithm provides better trade-off between the fairness demand and overall accuracy performance. 
