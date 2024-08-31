# TODO

1. Remove "optimal top k" from the table. Replace with the sum of all probabilities of children before truncation. 
2. Add a column with kl_divergencce between the distribution of children and the probability predicted by the model. 
3. Add a global field with the kl divergence between the tree and the joint distribution of the model. 
4. Implement pruning. 
5. Fix crashing bugs from reset.
6. Optimize the hell out of everything.  