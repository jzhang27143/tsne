# Parallelizing Gradient Calculations in t-SNE
By: Jason Zhang, Vy Nguyen

Our project video can be found here: https://www.youtube.com/watch?v=1XCk9WfPTds

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

## Milestone

So far, we have completed the basic infrastructure for our implementation, nearest neighbors calculations using FAISS, and perplexity estimation using binary search. We discovered that our disk quotas on the GHC machines do not allow us to install FAISS directly, so we resorted to creating a separate python script to dump nearest neighbor distances and indices in txt files. Since this is an initialization step, separating the t-SNE pipeline like this should not have a major impact on performance, but we will time the nearest neighbor calculations and try to find a GPU where we can leverage FAISS's CUDA optimizations.

One step that we overlooked when we created the proposal was determining how to initialize the variances of the Gaussian kernels. Official implementations set the variances so that the entropy is a fixed value e.g. 30 by default, but the variances that achieve this must be found using binary search. Thus, we created a CUDA kernel to conduct binary search for variance initialization.

Overall, we are on a good pace albeit slightly behind our proposed schedule. We estimate that we will be able to finish all components on time. Although we will most likely not be able to run CIFAR-10 experiments, we aim to perform K-means clustering on generated embeddings to get a quality measure. At the poster session, we intend to show the following:

* A scatter plot of the 2-D embeddings for MNIST showing distinct regions for each of the 10 digit classes
* A speedup plot showing speedups vs. the number of data points (we will run experiments using t-SNE with different size subsets of MNIST images)
* A speedup multiplier compared to SK-Learn's official t-SNE implementation (single-core).

At this point, there are not any major issues to deal with. Ideally, we would like to have a GPU that can run cuBLAS for our matrix-vector products (unfortunately, GHC machines do not have this library), but we are able to develop using thrust. 

## Schedule

* Week of March 28th

    * Integrate FAISS to produce nearest neighbor calculations for Iris dataset and start building the skeleton of the t-SNE code.

* Week of April 4th

    * Finish attractive force gradient calculations and begin Barnes-Hut algorithm implementation.
    
* Week of April 10th (Milestone report)

    * Implement a quad tree data structure (either from scratch or if an existing repo is usable) - Jason
    * Integrate the quad tree with the t-SNE code so that gradient calculations include both attractive and repulsive forces - Vy
    * At this point, the t-SNE implementation should be fully functional

* Week of April 17th

    * Benchmark our implementation at this point on Iris and MNIST - Jason
    * Explore different work allocation policies to improve load balancing - Jason, Vy
    * Implement utility to plot 2D embeddings for Iris and MNIST - Jason
    * Impelment k-Means clustering on embeddings and measure accuracy - Vy

* Week of April 23rd (Final report week)

    * Finalize results and write the final report - Jason, Vy

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
