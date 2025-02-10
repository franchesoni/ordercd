
# Unsupervised change detection for remote sensing

The idea is simple, use the temporal order to learn change detection. Monotonic changes indicate the passage of time (in any direction).
We want to see if we can learn to detect this from differences of DINOv2reg features. Therefore for any two images we compute their DINOv2reg feature difference per patch and work on that.

From the feature differences we get a sort of "change" feature, which we translate into a "distance" in time (that we argue can be only approximated by noting monotonic changes). 

This is:

feature maps:
$$f_1 = g(I_1),\; f_2 = g(I_2)$$
$$f_\delta = f_2 - f_1 \in \mathbb{R}^{F\times H\times W}$$ 

attention layer:
$$g(f_\delta) \in \mathbb{R}^{F\times H\times W}$$

spatial avg:
$$\bar{g}(f_\delta)=\frac 1 {HW} \sum_p g(f_\delta)[p]$$

linear + sigmoid is the score:
$$s_{12}=\text{sigmoid}(W^T\bar{g}(f_\delta))$$


Now the loss is kind of a double difference. We could try to order things by time and use an ordering loss, but we need to do it by looking at pairs. We don't ask the network to solve the ordering of time simply by the feature difference, as this is very hard to do (we suppose). In contrast, we only ask it to say when two features are apart in time. But how to do this without the dates? (we don't use the dates, only the sequence of ordered frames). We do know that images $(i,j)$ with $i<j$ are closer to each other than images $(i,k)$ with $j<k$. We ask the predicted distance to be greater for pairs that are more separated than other pairs (it's, in a way, a double difference). 
Formally, let $s_{b,i,j}$ correspond to estimated change score between images $i$ and $j$ of batch $b$. We define the set of greater index pairs for each reference $ i $ as  
$ \mathcal{P}_i = \{(j,k) : |i-j| > |i-k|\} $, the pairs that have distances greater than $|i-j|$.  
Then the loss function is given by  
$$
L = \frac{1}{B\cdot N} \sum_{b=1}^B \sum_{i=1}^T \sum_{(j,k) \in \mathcal{P}_i} \max\Bigl\{0,\; m - \Bigl(s_{b,i,j} - s_{b,i,k}\Bigr)\Bigr\},
$$
where  $ N = \sum_{i=1}^T |\mathcal{P}_i| $ and $m$ is a margin.  