---
layout: post
title:  "Multimodal Likelihood and Posterior"
date:   2017-08-06 20:40:00 +0800
permalink: /blog/multimodal
comments: true
categories: jekyll update
---



<p>It has long been thought that variational methods with factorized posterior have difficulty handling multimodal posterior. The fact that VAEs generate blurry images has been blamed on this inability, and much effort in the past two years has focused on supporting complex, multimodal posteriors [3] [4] [5]. In this post, we make the alternative proposition that multimodal posterior is not as bad a problem as we have thought, and propose that multimodal likelihood is a problem that deserves more attention.</p>

<p>We begin this discussion by first illustrating what multimodal posterior and multimodal likelihood actually “look like” in the image domain.</p>



<h2 id="multimodal-posterior">Multimodal Posterior</h2>

<p>If some image <script type="math/tex" id="MathJax-Element-926">x</script> has multimodal posterior <script type="math/tex" id="MathJax-Element-927">P(z|x)</script>, it means that there are multiple interpretations of <script type="math/tex" id="MathJax-Element-928">x</script> that conflict with each other. One type of image that quite certainly have multimodal posteriors are images involving some visual paradox. An example is given below:</p>

<p><img src="https://www.onelargeprawn.co.za/wp-content/uploads/2011/Shuplyak_faces_02.jpg" alt="enter image description here" title=""></p>

<p>In this image, we could interpret the houses as the eyes of a person. Thus it has several modes of conflicting interpretation. However, the point we would like to make here is that, <strong>this kind of image is not observed frequently</strong> in natural settings. Because this kind of image with complex interpretations do not occur frequently, in practice they probably do not pose as serious a problem as we think they do. </p>

<p>Instead, we would like to argue that, for methods involving learning by reconstruction, multimodal likelihood is a much more severe problem.</p>

<h2 id="multimodal-likelihood">Multimodal Likelihood</h2>

<p>If some latent code <script type="math/tex" id="MathJax-Element-959">z</script> is mapped to observation using a multimodal likelihood <script type="math/tex" id="MathJax-Element-960">P(x|z)</script>, there are multiple concrete realizations of <script type="math/tex" id="MathJax-Element-961">z</script> that are incompatible with each other. To illustrate this, let’s set <script type="math/tex" id="MathJax-Element-962">z=dog</script>, and create a few samples from a hypothetical likelihood <script type="math/tex" id="MathJax-Element-963">P(x|z)</script>.</p>

<p><img src="https://lh3.googleusercontent.com/-AsXvKYhVAZM/WXhh4bXjygI/AAAAAAAADhM/ehlGUQ3cR2ghjQbGo0E9XiB9TpF1zfvJACLcBGAs/s800/multimode_dog.png" alt="enter image description here" title="multimode_dog.png"></p>

<p>In the image above, a and b are both samples from <script type="math/tex" id="MathJax-Element-964">P(x|z)</script>, and b is a shifted version of a. Both a and b are likely under <script type="math/tex" id="MathJax-Element-965">P(x|z)</script>, because they look like real dog images. However, their mean, shown on the right, is unlikely under <script type="math/tex" id="MathJax-Element-966">P(x|z)</script>. This is the characteristic behaviour of multimodal distributions: the mean of multiple likely samples become unlikely.</p>

<p>In particular, this is also the observed problem with VAEs: they generate blurry images. In VAEs that do not employ multimodal likelihood, the same <script type="math/tex" id="MathJax-Element-967">z</script> is used to reconstruct multiple incompatible <script type="math/tex" id="MathJax-Element-968">x</script>, leading the generator to always reconstruct the mean, which creates the least likely result.</p>

<p>The image below are samples from DRAW [1], trained on CIFAR10. This demonstrates the typical case of VAEs generating blurry images.</p>

<p><img src="https://lh3.googleusercontent.com/5NsSqzOiXjfeBNi9GTYlJUuzwVye1TI-gOVp_gnSLqHWt13QLkbLgFQgwPqC4VicZhhL_p8zos79=s800" alt="enter image description here" title="DRAW_blur_samples.png"></p>

<p>Because the generative process of natural images often exhibit multimodal likelihood, in our models it is best if we accommodate this multimodality explicitly. In the next section, we’ll outline a method that has this potential.</p>

<h2 id="representing-multimodal-likelihood">Representing Multimodal likelihood</h2>

<p>The most trivial multimodal distribution is probably the categorical distribution. A categorical distribution with <script type="math/tex" id="MathJax-Element-840">K</script> categories encodes the probability of <script type="math/tex" id="MathJax-Element-841">K</script> different points simultaneously. However, samples from a categorical distribution are one-hot. This means that categorical distributions cannot efficiently represent “intensity”. For example, a categorical distribution cannot represent the difference between an image with strong contrast and another image with similar content but with much weaker contrast.</p>

<p><img src="https://lh3.googleusercontent.com/-ishOe7Vl0So/WXmwyM5Wp9I/AAAAAAAADiI/0PwwqCYlHssgE_5aqwgQeaaxhK7LdGEzQCLcBGAs/s800/strong_weak_dog.png" alt="enter image description here" title="strong_weak_dog.png"></p>

<p>The above figure contains two dog images. One with strong contrast and another with weak contrast. Samples from a categorical distribution will not be able to efficiently represent the difference between these two images.</p>

<p>To make up for this deficiency, we can multiply a categorical variable with a continuous variable, with the former representing category and the latter representing the intensity of the present category. </p>



<p><script type="math/tex; mode=display" id="MathJax-Element-842">\begin{align*}
\mathbf{m} &\sim P(\mathbf{m|{z}})   && \text{one-hot categorical sample} \\
\mathbf{s}  &\sim P(\mathbf{s|z})   && \text{continuous sample}  \\
\mathbf{x}  & = \mathbf{ms}  && \text{point-wise product of mask with continuous vector}
\end{align*}</script></p>

<p>We call this approach of multiplying a categorical mask with a continuous variable “Categorical-Times-Continuous”. It offers several advantages:</p>

<ol>
<li>It is inherently multimodal</li>
<li>It encodes intensity (categorical variables do not)</li>
<li>Samples of this distribution permit gradient propagation</li>
</ol>

<p>The image below is a graphical representation of the Categorical-Times-Continuous (CTC) approach. <script type="math/tex" id="MathJax-Element-843">m</script> is the categorical mask and <script type="math/tex" id="MathJax-Element-844">s</script> is the continuous vector.</p>

<p><img src="https://lh3.googleusercontent.com/-T0jgyFz8qHI/WXm7pYEiu9I/AAAAAAAADic/7KulZdbrQxwc3qIyXG4qeTFXKDYkPntNACLcBGAs/s800/ctc.png" alt="enter image description here" title="ctc.png"></p>

<p>CTC is closely related to ReLU, max pooling, and Local Winner Take All [2]. Next we briefly discuss these connections.</p>

<p>In the binary case, CTC can be related to ReLU, which is the product of a binary mask with a continuous value.</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-845">\begin{align*}
  f(x) &= mx \\ 
  m &= x > 0 
\end{align*}</script></p>

<p>ReLU is a deterministic function. It also bears some similarity to a zero-inflated Poisson that always samples the mean.</p>

<p>Max pooling, on the other hand, can also be expressed as the point-wise multiplication of a categorical mask with a continuous vector, followed by summation (omitted below).</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-1012">\begin{align*}
  f(\mathbf{x}) &= \mathbf{m}  \mathbf{x} && \text{point-wise multiply}
\\ \mathbf{x} &= [x_1, x_2, x_3, x_4] && \text{2x2 pooling has 4 inputs}
\\ \mathbf{m} &= (\mathbf{x} == max(\mathbf{x})) && \text{one-hot at maximum}
\end{align*}</script></p>

<p>In fact, part of the motivation in designing CTC is to replace ReLU and max pooling used in VAEs, so as to create a hierarchical model for images that is at the same time multimodal. To create the stochastic counterpart of ReLU, we replace the deterministic mask with a stochastic mask sampled from a Bernoulli, which is computed with a sigmoid. Similarly, in stochastic max pooling, the mask is sampled from a categorical distribution computed with a softmax.</p>

<p>This is a work-in-progress. We will provide more implementation details in the future. We have observed in experiments that CTC exhibits very large variance in learning. In the next post, we’ll discuss several types of variance that CTC is particularly sensitive to, along with several variance reduction techniques specifically developed for it.</p>

<h2 id="references">References</h2>

<p>[1] Karol Gregor, et al., DRAW: Recurrent Neural Network For Image Generation <br>
[2] Rupesh Srivastava, et al., Compete to Compute <br>
[3] Danilo Jimenez, et al., Variational Inference with Normalizing Flows <br>
[4] Diederik Kingma, et al., Improving Variational Inference with Inverse Autoregressive Flow <br>
[5] Tim Salimans, et al., Markov Chain Monte Carlo and Variational Inference: Bridging the Gap <br>
[6] Max Welling, et al., Bayesian Learning via Stochastic Gradient Langevin Dynamics</p>



<h2 id="gradient-descent-for-multimodal-posterior">Gradient Descent for Multimodal Posterior</h2>

<p>It seems inappropriate to talk about multimodal posteriors without mentioning this gradient descent method. But I really don’t know how to fit this section into the body of this post, so I’ll just put it after the References section. </p>

<p>In this tiny section, we briefly mention a method that allows pretty much any explicitly specified model to work with multimodal posteriors. Specifically, during inference, we start at a bad approximation given by the recognition model, and we use gradient descent on  <script type="math/tex" id="MathJax-Element-1167">log\frac{1}{P(x,z)}</script> w.r.t. latent state to drop into one of the modes of the true posterior. When this gradient descent settles, we’d have reached a local maximum of <script type="math/tex" id="MathJax-Element-1168">P(x,z)</script>, which is by definition a mode in the true posterior.</p>

<p>In the literature, this method has received various names. Predictive coding is one of such names. A more statistically principled variant is called MCMC with Langevin dynamics [6]. </p>
