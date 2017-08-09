---
layout: post
title:  "Variance reduction for Categorical-Times-Continuous"
date:   2017-08-09 20:00:00 +0800
permalink: /blog/ctc_variance
comments: true
categories: jekyll update
---


<p>In the <a href="/blog/multimodal">previous post</a>,  we discussed the issue of multimodal likelihood, and gave a brief introduction to the Categorical-Times-Continuous (CTC) scheme of representing multimodal distribution. This CTC scheme suffers from very high variance, which creates unstable learning. In this post, we discuss two types of variance that CTC is sensitive to, along with the corresponding methods of variance reduction.</p>

<p>Traditional latent variable models such as those using Gaussian or Laplacian posteriors are relatively straightforward to train. In comparison, latent variable models involving categorical variables are completely different. They exhibit an excessive amount of variance that makes training difficult. </p>



<h2 id="two-types-of-variance">Two types of variance</h2>

<p>Basically, the most common type of variance is input-sampling variance, which is a result of using different subsets of the dataset at every step of training. But this kind of variance is present everywhere and hasn’t been shown to be a big problem. So in this post we skip the discussion on this. Instead, we will focus on two types of variance that can be quite destructive when working with categorical variables:</p>

<ol>
<li>Latent-sampling variance</li>
<li>Reward variance</li>
</ol>



<h2 id="latent-sampling-variance">Latent-Sampling Variance</h2>

<p>Latent-sampling variance is a result of the one-hot activation of a categorical varaible. With <script type="math/tex" id="MathJax-Element-4561">K</script> categories, every category only learns about <script type="math/tex" id="MathJax-Element-4562">\frac{1}{K}</script> of the time. The image below illustrates this case. When <script type="math/tex" id="MathJax-Element-4563">z</script> is one-hot (indicated by the red unit), only the weights highlighted in red have gradient. In this scheme, gradient becomes a rare thing, and learning is slowed down as a result. In our experience, latent-sampling variance isn’t a big issue when we only have a few hundred latent categories. But if we ever want to scale to tens of thousands of categories, this is something we must pay attention to.</p>

<p><img src="https://lh3.googleusercontent.com/-WlLgkW4g4wU/WVMgtfNGGfI/AAAAAAAADeM/tkDQMzdxPvAdPRjGKO2gtzQlHXxrCrLlACLcBGAs/s800/decoder_gradient.png" alt="enter image description here" title="decoder_gradient.png"></p>

<h3 id="self-organizing-map">Self-Organizing Map</h3>

<p>In order to reduce this type of variance, we propose using a variant of Kohonen’s Self-Organizing Maps [1] that’s been adapted to work with gradient descent. We’ll begin this section with a brief overview of the ideas behind SOMs.</p>

<p>The Self-Organizing Map is an unsupervised learning method that does two things: <br>
1. it performs clustering <br>
2. it organizes the clusters on a low-dimensional manifold</p>

<p>For example, below we visualize a SOM with a <script type="math/tex" id="MathJax-Element-9183">16 \times 16</script> grid, trained on MNIST:</p>

<p><img src="https://lh3.googleusercontent.com/-jEv0OMI4FkQ/WYrmvCGNLFI/AAAAAAAADkw/XH1O-HxD9qwKdS-auj09J8u_c3huor6_ACLcBGAs/s800/samples_z_0_185.jpg" alt="enter image description here" title="samples_z_0_185.jpg"></p>

<p>This SOM is a neural net with exactly <script type="math/tex" id="MathJax-Element-9184">16 \times 16</script> hidden units. Every cell in the above grid visualizes the weight of its corresponding hidden unit.</p>

<p>The interesting thing about SOMs is that it identifies a <em>smooth</em> manifold using a set of <em>discrete</em> units. Being a discrete representation, it can be easily adapted to form a categorical distribution. And since the hidden units (clusters) are organized on a somewhat smooth neighbourhood, we can make use of this smoothness to speed up learning.</p>

<p>When a SOM is presented with a data sample, it first identifies a hidden unit whose weight is the closest to the data sample. We call this unit the “winner”. Then, the winner’s weight is adapted so as to move it a bit closer to the data sample. However, in addition to updating the weights of the winner, we also update the weights of the winner’s neighbours. Basically, if a unit is very close to the winner on the grid, then it would be updated more. If it is further away on the grid, it is updated less.</p>

<blockquote>
  <p>This procedure, being a clustering algorithm, naturally has some similarity with K-means and Gaussian mixtures. K-means is different in the sense that for every data sample, only the winner’s weights are updated. Gaussian mixtures, on the other hand, differs primarily in the way in which we compute the posterior.</p>
</blockquote>

<p>In our formulation, we implement SOM as a pooling (or smoothing) function without any trainable parameters. First, we map input <script type="math/tex" id="MathJax-Element-9185">\mathbf{x}</script> to a categorical distribution <script type="math/tex" id="MathJax-Element-9186">\mathbf{p}</script> using a nonlinear network. We then perform pooling within <script type="math/tex" id="MathJax-Element-9187">\mathbf{p}</script> to obtain a new categorical distribution <script type="math/tex" id="MathJax-Element-9188">\mathbf{P}</script>. The diagram below illustrates the flow of operations:</p>

<p><img src="https://lh3.googleusercontent.com/-5WVtrbwB4r8/WYryRfZijqI/AAAAAAAADlE/n-itqXCdRb0YPTdllxlQbVB89XEl6-s_ACLcBGAs/s800/pooling_som.png" alt="enter image description here" title="pooling_som.png"></p>

<p>Below are the steps of mapping input <script type="math/tex" id="MathJax-Element-9189">\mathbf{x}</script> to a latent sample <script type="math/tex" id="MathJax-Element-9190">\mathbf{z}</script>:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-9967">\begin{align*}
    \mathbf{s} &= f(\mathbf{x}) && \text{encode input to latent space}
\\ \mathbf{p} &= softmax(\mathbf{s})  &&  \text{probability of $i$th category}
\\ P_i &= \sum_j G(i,j) p_j && \text{pool probability using Gaussian kernel}
\\ S_i &= \sum_j G(i,j) s_j && \text{pool continuous value using Gaussian kernel}
\\ \mathbf{m} &\sim \mathbf{P}  && \text{take one-hot categorical sample}
\\ \mathbf{z} &= \mathbf{m}\mathbf{S} && \text{output categorical-times-continuous}
\end{align*}</script></p>

<blockquote>
  <p>Several things to note in the above: <br>
  1. For why we multiply <script type="math/tex" id="MathJax-Element-9968">\mathbf{m}</script> with <script type="math/tex" id="MathJax-Element-9969">\mathbf{S}</script>, please refer to the <a href="/blog/multimodal">previous post</a> about the motivation of Categorical-Times-Continuous. <br>
  2. <script type="math/tex" id="MathJax-Element-9970">\mathbf{s}</script> has <script type="math/tex" id="MathJax-Element-9971">K</script> dimensions, one per category. <br>
  3. <script type="math/tex" id="MathJax-Element-9972">\mathbf{m}</script> is <script type="math/tex" id="MathJax-Element-9973">K</script>-dimensional and is one-hot. It is the mask that indicates which category is sampled. <br>
  2. <script type="math/tex" id="MathJax-Element-9974">G(i,j)</script> is the Gaussian kernel that we use for pooling/smoothing. If categories are placed on a 2D neighbourhood, where the coordinate of the <script type="math/tex" id="MathJax-Element-9975">i</script>th category is represented as <script type="math/tex" id="MathJax-Element-9976">(i_x, i_y)</script>, its value is proportional to <script type="math/tex" id="MathJax-Element-9977">exp(-\frac{(i_x - j_x)^2+(i_y - j_y)^2}{\sigma})</script>. <script type="math/tex" id="MathJax-Element-9978">\sigma</script> is the neighbourhood size parameter which is decayed towards 0 with time. For more information about this neighbourhood idea, you may look up <a href="http://www.cs.bham.ac.uk/~jxb/NN/l16.pdf">this tutorial</a> by John Bullinaria.</p>
</blockquote>

<p>So how does this method reduce latent-sampling variance? Previously, with normal categorical variables, learning is one-hot. With this method, the set of units over which the winner takes input (the red and pink units in the middle layer under the Gaussian curve, in the image above) will all receive gradient. As a result, learning is no longer one-hot.</p>

<p>If we use a big pooling kernel by setting <script type="math/tex" id="MathJax-Element-9979">\sigma</script> to a big value, a lot of hidden units will learn together. However, the capacity of the network would be restricted due to excessive smoothing. In practice, we decay <script type="math/tex" id="MathJax-Element-9980">\sigma</script> towards 0, so that when training begins, a lot of units can learn together to speed up progress. Towards the end, when <script type="math/tex" id="MathJax-Element-9981">\sigma</script> approaches 0, the pooling operation becomes an identity map, and the network recovers its original capacity without smoothing.</p>

<h2 id="reward-variance">Reward Variance</h2>

<p>Reward variance is well-known in the reinforcement learning community. In illustrating this type of variance, we find it helpful to imagine a dog walking around in a neighbourhood looking for food (see image below). Some dogs will be happy as long as there is food, so they would wander around, sometimes getting one serving, sometimes getting two. While this kind of “wandering about” sounds like reasonable behaviour for animals, when we are training a model, we’d like it to settle at some point that maximizes reward. If the model instead keeps jumping around and never settle at a point, that’d be quite a headache for us.</p>

<p><img src="https://lh3.googleusercontent.com/-Dwm8gnSgjEM/WYqpjWGT7LI/AAAAAAAADkI/VktZ5BOy39IzDIxO8vPhxf_WMB2dYvCpQCLcBGAs/s800/dog_food.png" alt="enter image description here" title="dog_food.png"></p>

<p>Traditionally, this kind of reward variance is reduced using an input-dependent baseline method [2]. More recently, some authors have relaxed the categorical formulation, and have achieved some success using another distribution that approximates the categorical distribution [3] [4]. While [3] and [4] are quite interesting in their own right, they do not directly address the reward variance. We have reproduced their results and found evidence that their methods, too, suffer from significant reward variance.</p>

<blockquote>
  <p>In particular, the authors of [3] and [4] recommend that temperature be kept at 1. However, decaying temperature towards 0 is a necessary condition for their approximate distributions to approach the categorical distribution. When we do decay temperature towards zero, training again becomes unstable due to large reward variance (loss starts rising instead of falling).</p>
</blockquote>

<p>Instead of following the paths of [3] and [4], we find it more natural to reduce variance for CTC using an input-dependent baseline method. For discrete variables, baseline methods are unbiased. However, we use a Categorical-Times-Continuous formulation, which allows gradient propagation. This means that if we use a naive baseline, the method would become heavily biased. To counter this bias, we develop a bias correction term, which we document in the next section.</p>

<h3 id="input-dependent-baseline">Input-dependent baseline</h3>

<p>In this section, we give a specific implementation of the CTC scheme in the framework of VAEs, along with an input-dependent baseline method for variance reduction.</p>

<blockquote>
  <p>Notation:  <br>
  1. in this section, all lower-case letters are vectors. Typing all the bold letters is too tedious &gt;.&lt; <br>
  2. <script type="math/tex" id="MathJax-Element-10571">Q(x)</script> is the real data distribution <br>
  3. <script type="math/tex" id="MathJax-Element-10572">Q(z|x)</script> is the approximate posterior <br>
  4. <script type="math/tex" id="MathJax-Element-10573">P(x,z)</script> is the generative model <br>
  4. Summations imply monte carlo integration</p>
</blockquote>

<p>The objective we minimize in a VAE is: <br>
<script type="math/tex; mode=display" id="MathJax-Element-10574">
\sum_x Q(x) \sum_z Q(z|x) log\frac{Q(z|x)}{P(x,z)}
</script> <br>
We compute the latent code <script type="math/tex" id="MathJax-Element-10575">z</script> using the following procedure:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-10770">\begin{align*}  
           s =& f_\theta(x)
\\ Q(m|x) =& softmax(s)          && \text{compute categorical distribution}
\\ m \sim & Q(m|x)                 && \text{take one-hot sample}
\\ z       =& ms                         && \text{categorical-times-continuous}
\end{align*}</script></p>

<blockquote>
  <p>Note: <script type="math/tex" id="MathJax-Element-10771">s</script> is actually a Gaussian sample, but we omit that for clarity.</p>
</blockquote>

<p>We focus on the gradient from the generative model, and pathwise gradient w.r.t. <script type="math/tex" id="MathJax-Element-10772">\theta</script> is:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-13665">\begin{align*}
     & \nabla_\theta \sum_x Q(x) \sum_z Q(z|x) log\frac{1}{P(x,z)}
\\=& - \sum_{x,z} Q(x,z) \nabla_\theta logP(x,z)
     &&\text{only take pathwise gradient}
\\=& -\sum_{x,z} Q(x,z) \nabla_z logP(x, z) \frac{\partial z}{\partial \theta}
\\=& -\sum_{x,z} Q(x,z) (\frac{\partial z}{\partial \theta})^T (\nabla_z logP(x, z))^T
     &&\text{transpose to improve presentation}
\end{align*}</script></p>

<p>We subtract a baseline in the gradient: <br>
<script type="math/tex; mode=display" id="MathJax-Element-13666">
 -\sum_{x,z} Q(x,z) (\frac{\partial z}{\partial \theta})^T ((\nabla_z logP(x, z))^T - b(x,z))
</script></p>

<p>This introduces a bias in the gradient: <br>
<script type="math/tex; mode=display" id="MathJax-Element-13667">
 -\sum_{x,z} Q(x,z) (\frac{\partial z}{\partial \theta})^T  (-b(x,z))
</script></p>

<p>We can correct the bias by adding the following into the update: <br>
<script type="math/tex; mode=display" id="MathJax-Element-13668">\begin{align*}
     & - \sum_{x,z} Q(x,z) (\frac{\partial z}{\partial \theta})^T b(x,z)
\end{align*}</script></p>

<p>We maintain an adaptive estimate of <script type="math/tex" id="MathJax-Element-13669">\omega = \sum_{x,z} Q(x,z) (\frac{\partial z}{\partial \theta})^T  b(x,z)</script> using an exponential moving average of samples from the minibatches.</p>

<p>Note that in practice, we optimize with <script type="math/tex" id="MathJax-Element-13670">M</script> minibatches, each with <script type="math/tex" id="MathJax-Element-13671">N</script> samples, so the update (without learning rate) is: <br>
<script type="math/tex; mode=display" id="MathJax-Element-13672">\begin{align*}
     & \sum_{m=1}^M \left[ \frac{1}{N}\sum_{x \in m}  \sum_z Q(z|x) (\frac{\partial z}{\partial \theta})^T ((\nabla_z logP(x, z))^T - b(x, z)) \right] 
        +\frac{MN}{N}\omega
\\=&\sum_{m=1}^M \left[ \frac{1}{N}\sum_{x \in m} \sum_z Q(z|x) (\frac{\partial z}{\partial \theta})^T ((\nabla_z logP(x, z))^T - b(x,z)) 
        + \omega \right] 
\end{align*}</script></p>

<h2 id="experimental-results">Experimental Results</h2>

<p>TBD. Will patch up this section soon.</p>

<h2 id="conclusion">Conclusion</h2>

<p>In this post we’ve explained two types of variance that plague the CTC coding scheme. To handle latent-sampling variance, we introduced a variant of Self-Organizing Maps, which removes one-hot learning through a pooling operation. To handle reward variance, we developed an input-dependent baseline method with a bias correction term. We’ll patch this post with more recent result soon.</p>

<p>For the next post, I have two things in mind: <br>
1. A more broadly applicable variance reduction technique using a critic <br>
2. An image model that incorporates the CTC scheme into a convolutional backbone</p>

<p>So the next two posts are likely to be about these topics.</p>

<h2 id="references">References</h2>

<p>[1] Teuvo Kohonen, The Self-organizing map <br>
[2] Andriy Mnih, et al., Neural Variational Inference and Learning in Belief Networks <br>
[3] Eric Jang, et al., Categorical Reparameterization with Gumbel-Softmax <br>
[4] Chris Maddison, et al., The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables</p>
