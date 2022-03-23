# Parallelizing Gradient Calculations in t-SNE
By: Jason Zhang, Vy Nguyen

## Summary
We plan to parallelize the t-distributed Stochastic Neighbor Embedding (t-SNE) method for non-linear dimensionality reduction using NVIDIA GPU's. Sequential implementations have seen speedups in algorithmic complexity from nearest-neighbors and Barnes-Hut approximations [1], and in this work we seek to exploit parallelism to further accelerate the gradient calculations in t-SNE.

## Background
Given a high-dimensional dataset with N data points, the objective of t-SNE is to learn a low-dimensional embedding (usually 2 or 3 dimensions to allow visualization.) t-SNE achieves this by keeping the same relative distances between all pairs of points across the high and low-dimensional spaces. More specifically, t-SNE computes a distribution p corresponding to the pair-wise distances between all pairs of high-dimensional points and a similar distribution q across all pairs of embedding vectors. t-SNE learns the embeddings by using gradient descent to minimize the KL-Divergence between p and q [2].

A naive implementation of t-SNE has quadratic complexity with respect to the dataset size N from considering distances between all pairs of points, which is infeasible for most datasets. However, [1] shows that the gradient calculations can be treated as an N-body physics problem wherein each embedding's gradient is a combination of attractive and repulsive forces. The attractive forces for one point's embedding can be approximated by only considering contributions from its nearest neighbors. Furthermore, repulsive forces can be efficiently computed using a quad-tree as in the Barnes-Hut algorithm [3]. 

The gradient calculations are the most intensive component of t-SNE, and there is a clear opportunity for parallelism across the N embedding points since each embedding point's gradients can be computed independently during an iteration of gradient descent. 

## Challenges

As mentioned above, computing the embedding gradients has two components: attractive and repulsive forces. The attractive forces require identifying nearest neighbors for each point before gradient descent, but during gradient descent, the neighbor points are fixed so there is a high degree of memory locality for the attractive forces. This makes the attractive forces an easier component to parallelize than the repulsive forces, and after taking time constraints into account, we choose to concentrate our parallelization efforts on the repulsive forces. Computing the nearest neighbors is a difficult problem in its own, but we can leverage fast similarity search engines such as FAISS [4] to precompute the nearest neighbors before gradient descent.

In contrast, the repulsive forces are much harder to compute, requiring the data points to be organized in a quad-tree as per the Barnes-Hut algorithm. While we can theoretically parallelize repulsive forces by simply assigning embedding vectors to processors, the main challenge to achieve high speedups is that the amount of work per node in the quad-tree of Barnes-Hut algorithm is highly **non-uniform** since the depths of different points in the tree are irregular. This may also cause high degrees of memory contention. Since the embedding space will evolve as the gradients are updated, parallelization over the gradient calculations will require smart work allocations, including the use of semi-static assignment and spatial locality.

Through this project, we hope to learn more about parallelization in Barnes-Hut approximation, and the importance of a good work allocation policy in a challenging parallel task.

## Resources

To implement nearest neighbor search for the attractive forces before gradient descent, we plan to use an existing code so we can focus on parallelization for the repulsive forces. In particular, we will utilize FAISS[^1], which is highly optimized for CUDA [4].

For the remainder of our code, we plan to start from scratch, using the code base provided by [1] as a reference for the sequential implementation of t-SNE[^2]. We will also refer to [5] as a resource for optimization of the Barnes-Hut algorithm on GPU's.

## Goals and Deliverables

* 100% plan

    * Fully develop a parallelized implementation of t-SNE with Nearest-Neighbor and Barnes-Hut approximations.
 
    * Run the gradient descent algorithm and provide a 2D visualization of class clustering on Iris and MNIST datasets.
 
    * Provide a speedup comparison between the sequential implementation (SK-Learn) and our parallel version. We are not sure what speedup to expect, but conservatively we will aim for at least 50x based on similar works [6].

* 125% plan extra goals

    * Provide a 2D visualization of class clusters on CIFAR-10 data set.
    
    * Perform K-means clustering on the Iris, MNIST, and CIFAR-10 embeddings to get accuracy measures for the embeddings.

* 75% plan

    * Same as 100% plan but work assignment policy for Barnes-Hut may be naive. 

## Platform Choice

We plan to use CUDA to implement the parallelization version and collect results on NVIDIA GPU's. FAISS is highly optimized for CUDA, related works have parallized t-SNE using CUDA, and we may expect to achieve greater speedups by taking advantage of the warp architecture.

## Schedule

* Week of March 28th

    * Integrate FAISS to produce nearest neighbor calculations for Iris dataset and start building the skeleton of the t-SNE code.

* Week of April 4th

    * Finish attractive force gradient calculations and begin Barnes-Hut algorithm implementation.
    
* Week of April 10th (Milestone report)

    * Finish first implementation of t-SNE on CUDA (most likely with sub-optimal work allocation) and write milestone report.

* Week of April 17th

    * Explore different work allocation policies to improve load balancing and benchmark speedups on Iris and MNIST datasets.

* Week of April 23rd (Final report week)

    * Finalize results and write the final report.

## References
[1] L. Van Der Maaten. 2014. Accelerating t-SNE using tree-based algorithms. J. Mach. Learn. Res. 15, 3221–3245.

[2] L. Van Der Maaten and G. Hinton. 2008. Visualizing Data using t-SNE. J. Mach. Learn. Res. 9, 2579--2605.

[3] J. Barnes and P. Hut. 1986. A hierarchical O(N log N) force-calculation algorithm. Nature, 324
(4):446–449.

[4] J. Johnson, M. Douze, and H. Jegou. 2017. Billion-scale similarity search with GPUs

[5] M. Burtscher and K. Pingali, “An efficient cuda implementation of the tree-based barnes hut n-body algorithm,” in GPU computing Gems Emerald edition. Elsevier, 2011, pp. 75–92.

[6] D. M. Chan, R. Rao, F. Huang, J. F. Canny : t-SNE-CUDA: GPU-Accelerated t-SNE and its Applications to Modern Data

[^1]: https://github.com/facebookresearch/faiss
[^2]: https://github.com/lvdmaaten/bhtsne
