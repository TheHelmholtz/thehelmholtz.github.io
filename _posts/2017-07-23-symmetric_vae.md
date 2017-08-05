---
layout: post
title:  "Symmetric variational loss and GAN"
date:   2017-08-05 10:00:00 +0800
permalink: /blog/symmetric_vae
comments: true
excerpt: We discuss the asymmetry of Variational Autoencoders, and why it might be better to use a symmetric cost instead. We show several symmetric variational costs, and demonstrate how GANs can be derived from them. We have succeeded in training with one of such costs, and demonstrate some sample images.
categories: jekyll update
---








<h2 id="abstract">Abstract</h2>

<p>Variational Autoencoders optimize an asymmetric cost, which is undesirable. We extend VAEs to a symmetric cost, which is the sum of two Minimum Description Length (MDL) objectives. This MDL objective has two problems. First, it cannot be directly computed. In approximating it we obtain the discriminator found in Generative Adversarial Networks. Second, it involves two pairs of opposing costs, which makes optimization difficult. We break the opponency by minimizing an upper bound on the MDL objective, obtaining the Extended Helmholtz Machine (EHM). We demonstrate early results of an EHM trained on LSUN bedroom images.</p>

<h2 id="vaes-asymmetry">VAE’s asymmetry</h2>

<blockquote>
  <p>Notation: <script type="math/tex" id="MathJax-Element-3662">Q(x)</script> is the real data distribution. <script type="math/tex" id="MathJax-Element-3663">Q(z|x)</script> is the approximate posterior. <script type="math/tex" id="MathJax-Element-3664">P(z)</script> is the prior over the latent code and <script type="math/tex" id="MathJax-Element-3665">P(x|z)</script> is the likelihood of the generative model. Also, summations such as <script type="math/tex" id="MathJax-Element-3666">\sum_{x,z}Q(x,z)</script> imply Monte Carlo integration, so gradient is not taken on these terms. <strong>I’ve used summation instead of expectation</strong> to make my LaTeX code readable. Also, all lower-case letters are vectors.</p>
</blockquote>

<p>The objective of VAE [9] is:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-3070">\begin{align*}
     &\sum_x Q(x) \sum_z Q(z|x)log\frac{Q(z|x)}{P(x,z)}
\\=&\sum_x Q(x)[KLD( Q(z|x) || P(z|x) ) + log\frac{1}{P(x)}]
\end{align*}</script></p>

<p>VAE is asymmetric in the sense that sampling is performed in only one direction (from <script type="math/tex" id="MathJax-Element-3071">Q</script>). In comparison, both Boltzmann Machine [10] and Helmholtz Machine [11] perform sampling from both <script type="math/tex" id="MathJax-Element-3072">Q</script> (real data) and <script type="math/tex" id="MathJax-Element-3073">P</script> (fantasy data). The disadvantage of sampling from only <script type="math/tex" id="MathJax-Element-3074">Q</script> is that there will be regions in <script type="math/tex" id="MathJax-Element-3075">(x,z)</script> whose probability under <script type="math/tex" id="MathJax-Element-3076">Q</script> is zero, yet whose probability under <script type="math/tex" id="MathJax-Element-3077">P</script> is greater than zero.</p>

<p><img src="https://lh3.googleusercontent.com/-G6nSM2ag-oU/WVe04A09biI/AAAAAAAADgw/SP8_NeS-vRE6Ckyf-hIGn-B4YBxHfVDggCLcBGAs/s0/Q_and_P.png" alt="enter image description here" title="Q_and_P.png"></p>

<p>The image above illustrate this possibility. Because training is performed by sampling exclusively from <script type="math/tex" id="MathJax-Element-3078">Q</script>, the regions that are not covered by <script type="math/tex" id="MathJax-Element-3079">Q</script> may misbehave under <script type="math/tex" id="MathJax-Element-3080">P</script>.</p>

<blockquote>
  <p>An intuitive interpretation: consider the case where a teacher is teaching a student to solve some problems. The teacher has limited time, so he only explains a subset of all possible problems. Let’s call the problems that the teacher has shown to the student the “taught problems”, and the rest “untaught problems”. </p>
  
  <p>If the student blindly accepts whatever the teacher teaches, he will not be able to handle untaught problems that are different from the taught ones. However, if the student is very curious, and frequently asks questions, and the teacher in turn gives answers, then we can reasonably expect this curious student to be much better at solving the untaught problems.</p>
  
  <p>When sampling from <script type="math/tex" id="MathJax-Element-3081">Q</script>, the teacher is teaching the “taught problems”. When sampling from <script type="math/tex" id="MathJax-Element-3082">P</script>, the student is asking questions about “untaught problems”.</p>
</blockquote>

<h2 id="symmetric-vae">Symmetric VAE</h2>

<p>In the Symmetric VAE, the objective has two parts. The first part is the original VAE objective, with sampling performed from <script type="math/tex" id="MathJax-Element-3667">Q</script>. The second part is its symmetric form, with sampling performed from <script type="math/tex" id="MathJax-Element-3668">P</script>. It is written below:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-3121">\begin{align*}
&\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(x,z)}
\end{align*}</script></p>

<p>This objective is an upper bound on the <script type="math/tex" id="MathJax-Element-3122">-logP(x)</script> and <script type="math/tex" id="MathJax-Element-3123">-logQ(z)</script> (plus two KLDs), which is exactly what we want. Unfortunately, we cannot optimize the above objective directly, because it involves <script type="math/tex" id="MathJax-Element-3124">Q(x,z)=Q(x)Q(z|x)</script>, which can be sampled from but cannot be computed (recall <script type="math/tex" id="MathJax-Element-3125">Q(x)</script> is the data distribution). To optimize the above objective, we need to approximate <script type="math/tex" id="MathJax-Element-3126">Q(x)</script>, and replace <script type="math/tex" id="MathJax-Element-3127">Q(x)</script> with its approximator. In doing this, we will obtain the discriminator of GANs [12].</p>

<h3 id="approximating-qx">Approximating <script type="math/tex" id="MathJax-Element-3669">Q(x)</script></h3>

<p>One method to approximately compute the Symmetric VAE objective is to use an approximate <script type="math/tex" id="MathJax-Element-3670">Q(x)</script>. We can train a normalized probability <script type="math/tex" id="MathJax-Element-3671">Q'(x)</script> to approximate <script type="math/tex" id="MathJax-Element-3672">Q(x)</script> by minimizing the cross entropy: <br>
<script type="math/tex; mode=display" id="MathJax-Element-3673">\begin{align*}
      &\sum_x Q(x)log\frac{1}{Q'(x)}
\end{align*}</script></p>

<p>When <script type="math/tex" id="MathJax-Element-3674">Q'(x)</script> and <script type="math/tex" id="MathJax-Element-3675">Q(x)</script> are identical, we can recover the original Symmetric VAE objective exactly.</p>

<p>One complication with the above approach is that training with <script type="math/tex" id="MathJax-Element-3676">Q'(x)</script>, a normalized probability, is difficult. So instead we train with <script type="math/tex" id="MathJax-Element-3677">Q''(x)</script>, an unnormalized probability of <script type="math/tex" id="MathJax-Element-3678">Q'(x)</script> <br>
<script type="math/tex; mode=display" id="MathJax-Element-3679">
\frac{1}{Z} Q''(x) = Q'(x)
</script> <br>
where <script type="math/tex" id="MathJax-Element-3680">Z=\sum_x Q''(x)</script> is the partition function. </p>

<p>Now, to approximate <script type="math/tex" id="MathJax-Element-3681">Q(x)</script>, we train <script type="math/tex" id="MathJax-Element-3682">Q''(x)</script> to minimize:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-2955">\begin{align*}
     & \sum_x Q(x) log\frac{1}{Q'(x)}
\\=& \sum_x Q(x) [log\frac{1}{Q''(x)} + logZ]
\end{align*}</script></p>

<p>When we expand <script type="math/tex" id="MathJax-Element-2956">Z</script>, and remove a term that’s irrelevant to the minimization, we obtain:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-3614">
E_{v \sim Q(x)}log\frac{1}{Q''(x)} - E_{x \sim P(x)}log\frac{1}{Q''(x)}
</script></p>

<p>which is the discriminator loss in Generative Adversarial Networks. So <script type="math/tex" id="MathJax-Element-3615">Q''(x)</script> is the discriminator. Please refer to Appendix A for details of derivation.</p>

<h2 id="extended-helmholtz-machine">Extended Helmholtz Machine</h2>

<p>After some experiments, we found it extremely difficult to optimize a Symmetric VAE, possibly because of the opponency that exists in the wake phase and the sleep phase. </p>

<p><script type="math/tex; mode=display" id="MathJax-Element-2117">
\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(z)P(x|z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(x)Q(z|x)}
</script></p>

<p>Specifically, the wake phase objective minimizes <script type="math/tex" id="MathJax-Element-2118">logQ(z|x)</script>, yet the sleep phase maximizes it. Also, the wake phase objective maximizes <script type="math/tex" id="MathJax-Element-2119">logP(x|z)</script>, while the sleep phase minimizes it.</p>

<p>If we remove the opponency, we get the objective of the Extended Helmholtz Machine, which minimizes two cross-entropies:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-2185">\begin{align*}
     &\sum_{x,z}Q(x,z)log\frac{1}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{1}{Q(x,z)}
\end{align*}</script></p>

<p>This objective is equivalent to the original Symmetric VAE objective, plus two entropy terms <script type="math/tex" id="MathJax-Element-2186">\sum_{x,z} Q(x,z)log\frac{1}{Q(z|x)}</script> and <script type="math/tex" id="MathJax-Element-2187">\sum_{x,z} P(x,z)log\frac{1}{P(x|z)}</script>.</p>

<h3 id="results-of-ehm">Results of EHM</h3>

<p>Here we provide some very early results of training the EHM. The images below are sampled from an EHM whose encoder, decoder and discriminator have structures similar to DCGAN [8]. This model is trained on 64x64 LSUN bedroom images. </p>

<p><img src="https://lh3.googleusercontent.com/-CALIt-h6MkQ/WX84FF62ruI/AAAAAAAADjU/clpwWghMihY4-Xz0SlqAvqdfaCDUA3xVQCLcBGAs/s800/ehm7_samples_01_014000.png" alt="enter image description here" title="ehm7_samples_01_014000.png"></p>

<p>Note that some of the samples contain a kind of strange artifact (last column of row 4 and 5). This kind of artifact has been observed in other GAN methods before, and we are working on understanding such artifacts.</p>

<p>After training the EHM, the wake-phase reconstruction loss is quite low, yet the sleep-phase reconstruction cost is high. Apparently the moment when the decoder starts to generate sharp images, the sleep-phase reconstruction cost starts rising. The implication of this is not yet understood. One possibility is that most of the latent dimensions are not utilized. As a result, there is no mutual information between those unutuilized dimensions and the generated image. Therefore, those dimensions could not be recovered from the image.</p>

<p>Another problem with the current EHM formulation is that, when we feed the encoder with a real image, and ask the generator to reconstruct it, the reconstruction error is low, and we get a rather blurry reconstruction. [TODO: reconstruction samples]. These reconstructed samples look nothing like samples directly obtained by sampling from <script type="math/tex" id="MathJax-Element-3683">P(z)</script> then mapping it through the decoder, suggesting that <script type="math/tex" id="MathJax-Element-3684">Q(z|x)</script> does not produce posterior samples that are likely under the prior.</p>

<blockquote>
  <p>We found it useful to tie the weights between the encoder and the decoder, which significantly speeds up training, and makes training a lot more stable. In our experiments, both the encoder and decoder are fully convolutional. The decoder, which consists of a series of transposed convolution, uses the transposed weights of the encoder. </p>
  
  <p>This weight sharing scheme is motivated by predictive coding [4][3]. Since backpropagation uses transposed weights, the encoder step can be considered as “guided backpropagation” [5] in minimizing a reconstruction cost. Details to be added.</p>
</blockquote>

<p><br><br><br><br><br><br></p>

<hr>

<p><strong>Under construction</strong>: the discussion and conclusion sections are far from complete. I’m still working on them.</p>

<p><br><br><br><br><br><br></p>

<h2 id="discussion">Discussion</h2>

<p>Currently we have several outstanding problems with the EHM:</p>

<ol>
<li>Generated images have significant artifacts</li>
<li>Sleep-phase reconstruction cost remains high</li>
<li>Wake-phase reconstructed samples remain blurry</li>
</ol>

<p>For problem one, there might be several possible explanations: </p>

<ol>
<li>Artifacts are introduced by wake-phase reconstruction, which forces the generator to enter strange locations (unlikely, since these artefacts have been observed in other GAN methods).</li>
<li>Artifacts are introduced by excessive gradient from the discriminator, which leaves training far from convergence. This can be suppressed if we apply weight clipping as in WGAN [1]. However, because generator receives gradient from several sources, once we weaken the gradient from the discriminator, it may no longer be trained towards the direction we want. Therefore, we cannot naively clip the discriminator weights. Several alternative solutions exist, one is to decay learning rate after every epoch by, say, half. Another is to put a hard, but carefully chosen, limit on gradient norm. Yet another is to penalize gradient norm [6].</li>
<li>Artifacts have something to do with batch normalization. Some authors have suggested replacing BN with weight normalization [7], and some suggested using layer normalization instead [6].</li>
</ol>

<p>For problem two, explanations include:</p>

<ol>
<li>generator has very steep surface. Small change in latent code leads to large change in generated images</li>
<li>generator relies on discontinuity to generate sharp images, and inverting such a highly nonsmooth function is hard</li>
<li>most dimensions in the latent code are not utilized. Naturally we cannot reconstruct noise.</li>
</ol>

<p>For problem three, we cannot rely on the posterior obtained by a single pass to do reconstruction. Also, after several stages of spatial down-sampling, it is difficult to imagine how the reconstruction can stay sharp. Instead, we need to infer the posterior by optimization (as in deconv nets [3]), then perform reconstruction using the inferred posterior.</p>

<p>We are still actively working on understanding and solving the three problems above.</p>

<h2 id="conclusion">Conclusion</h2>

<p>Currently with so many outstanding problems, we cannot properly conclude yet. Here we briefly reiterate several important points:</p>

<ol>
<li>We argue that probability models should be trained with symmetric sampling</li>
<li>We showed that GAN can be derived from the sleep-phase of a Symmetric VAE</li>
<li>We showed that the Extended Helmholtz Machine can be trained, but there are still many problems.</li>
</ol>

<p>A proper conclusion is TBD.</p>

<p>A discussion on using the discriminator as “image prior” is TBD.</p>

<h2 id="references">References</h2>

<p>[1] Martin Arjovsky, et al., Wasserstein GAN <br>
[2] Max Welling, et al., Bayesian Learning via Stochastic Gradient Langevin Dynamics <br>
[3] Matthew Zeiler, et al., Adaptive Deconvolutional Networks for Mid and High Level Feature Learning <br>
[4] Rafai Bogacz, A tutorial on the free-energy framework for modelling perception and learning <br>
[5] Jost Springenberg, et al., Striving for Simplicity: The All Convolutional Net <br>
[6] Ishaan Gulrajani, et al., Improved Training of Wasserstein GANs <br>
[7] Sitao Xiang, et al., On the Effects of Batch and Weight Normalization in Generative Adversarial Networks <br>
[8] Alec Radford, et al., Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks <br>
[9] Diederik Kingma, et al., Auto-Encoding Variational Bayes <br>
[10] David Ackley, et al., A Learning Algorithm for Boltzmann Machines <br>
[11] Peter Dayan, et al., The Helmholtz Machine <br>
[12] Ian Goodfellow, et al., Generative Adversarial Networks</p>

<p><br><br><br><br><br><br></p>

<hr>

<p><br><br><br><br><br><br></p>

<h2 id="appendix">Appendix</h2>

<h3 id="a-discriminator-loss">A. Discriminator loss</h3>

<blockquote>
  <p>Note: summation implies Monte Carlo integration, as indicated at the beginning of this article.</p>
</blockquote>

<p>We train an unnormalized probability <script type="math/tex" id="MathJax-Element-3538">Q''(x)</script> to match <script type="math/tex" id="MathJax-Element-3539">Q(x)</script>. We begin by converting <script type="math/tex" id="MathJax-Element-3540">Q''(x)</script> to a normalized form  <br>
 <script type="math/tex; mode=display" id="MathJax-Element-3541">Q'(x)=\frac{1}{Z}Q''(x)</script></p>

<p>where <script type="math/tex" id="MathJax-Element-3542">Z=\sum_x Q''(x)</script> is the partition function. It can be estimated using samples:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-3311">\begin{align*}
Z =& \sum_x Q''(x)
\\=& \sum_x P(x|z) \frac{Q''(x)}{P(x|z)} && \text{Resample. Equality holds for any $z$}
\\=& \sum_z P(z) \sum_x P(x|z) \frac{Q''(x)}{P(x|z)} &&\text{Sum over $z$}
\\=& \sum_{x,z} P(x,z) \frac{Q''(x)}{P(x|z)}
\end{align*}</script></p>

<p>This estimator is in fact sampling from <script type="math/tex" id="MathJax-Element-3312">P(x)</script>, but does not require us to estimate <script type="math/tex" id="MathJax-Element-3313">P(x)</script> itself (as is required in the simpler estimator of <script type="math/tex" id="MathJax-Element-3314">Z=\sum_x P(x) \frac{Q''(x)}{P(x)}</script>).</p>

<p>Now, to approximate <script type="math/tex" id="MathJax-Element-3315">Q(x)</script>, we train <script type="math/tex" id="MathJax-Element-3316">Q''(x)</script> to minimize the cross entropy from <script type="math/tex" id="MathJax-Element-3317">Q(x)</script> to <script type="math/tex" id="MathJax-Element-3318">Q'(x)</script>:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-3511">\begin{align*}
     & \sum_x Q(x) log\frac{1}{Q'(x)}
\\=& \sum_x Q(x) [log\frac{1}{Q''(x)} + logZ]
\end{align*}</script></p>

<p>Now we will expand <script type="math/tex" id="MathJax-Element-3512">Z</script> to its estimator. Note that <script type="math/tex" id="MathJax-Element-3513">(x,z)</script> in <script type="math/tex" id="MathJax-Element-3514">Z</script> are sampled from <script type="math/tex" id="MathJax-Element-3515">P</script>, and is completely independent from the <script type="math/tex" id="MathJax-Element-3516">x</script> samples taken from <script type="math/tex" id="MathJax-Element-3517">Q</script>. To reflect this independence, we replace the <script type="math/tex" id="MathJax-Element-3518">x</script> sampled from <script type="math/tex" id="MathJax-Element-3519">Q(x)</script> with <script type="math/tex" id="MathJax-Element-3520">v</script>. The above becomes: <br>
<script type="math/tex; mode=display" id="MathJax-Element-3521">\begin{align*}
     &\sum_v Q(v) [log\frac{1}{Q''(v)} + logZ]
\\=&\sum_v Q(v)\left[ log\frac{1}{Q''(v)} + log\sum_{x,z}P(x,z)\frac{Q''(x)}{P(x|z)} \right]
\\=&\sum_v Q(v)  log\frac{1}{Q''(v)}   - log\sum_{x,z}P(x,z)\frac{P(x|z)}{Q''(x)}
\\\le &\sum_v Q(v) log\frac{1}{Q''(v)}  - \sum_{x,z}P(x,z) log\frac{P(x|z)}{Q''(x)}
\end{align*}</script></p>

<p>Because we only minimize the above w.r.t. parameters of <script type="math/tex" id="MathJax-Element-3522">Q''(x)</script>, it is equivalent to minimizing:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-3559">
\sum_v Q(v) log\frac{1}{Q''(v)} - \sum_x P(x) log\frac{1}{Q''(x)}
</script></p>

<p>This is the discriminator loss minimized by WGAN [1].</p>

<h3 id="b-generator-gradient">B. Generator Gradient</h3>

<p>For the sake of demonstrating VAE’s connection with GANs, we can also obtain a gradient term similar to the generator gradient in GANs. Since GANs do not have the inference network <script type="math/tex" id="MathJax-Element-2926">Q(z|x)</script>, we remove all components of <script type="math/tex" id="MathJax-Element-2927">Q(z|x)</script>, and replace <script type="math/tex" id="MathJax-Element-2928">Q(x)</script> with <script type="math/tex" id="MathJax-Element-2929">Q'(x)</script>:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-2674">\begin{align*}
     &\sum_z P(z) \sum_x P(x|z)log\frac{P(x|z)}{Q'(x)}
\end{align*}</script></p>

<p>We differentiate <script type="math/tex" id="MathJax-Element-2675">log\frac{P(x|z)}{Q'(x)}</script> against the decoder parameter <script type="math/tex" id="MathJax-Element-2676">\phi</script> (the summations are Monte Carlo integration):</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-3657">\begin{align*}
     &\nabla_\phi log\frac{P(x|z)}{Q'(x)}
\\=&  \nabla_\phi log\frac{1}{Q'(x)} 
       - \nabla_\phi log\frac{1}{P(x|z)}
\end{align*}</script></p>

<p>Note that in GANs, <script type="math/tex" id="MathJax-Element-3658">P(x|z)=1</script>, but it isn’t the case for VAEs, which is why we have the second term above. The first term in the above can be written as: <br>
<script type="math/tex; mode=display" id="MathJax-Element-3659">\begin{align*}
     & \nabla_x log \frac{1}{Q'(x)} \frac{\partial x}{\partial \phi}
\\=&[\nabla_x log \frac{1}{Q''(x)} + \nabla_x logZ]  \frac{\partial x}{\partial \phi}
\\=&\nabla_x log \frac{1}{Q''(x)} \frac{\partial x}{\partial \phi}
     && \text{$Z$ does not depend on $x$}
\end{align*}</script></p>

<p>Which is the generator gradient. The above also suggests that the unnormalized probability <script type="math/tex" id="MathJax-Element-3660">Q''(x)</script> can be used directly to replace <script type="math/tex" id="MathJax-Element-3661">Q(x)</script> in optimization.</p>

<p><br><br><br><br><br><br></p>

<hr>

<p><br><br><br><br><br><br></p>

<h2 id="extras-do-we-really-need-an-external-discriminator">Extras: Do we really need an external discriminator?</h2>

<p>Autoregressive models can generate sharp images without the use of a discriminator. This suggests that, under the right conditions, it is possible to get rid of the discriminator and still get high-quality models. But how do we do that with a latent variable model?</p>

<ul>
<li>Would multimodal likelihood and multimodal posterior help?</li>
<li>Would full-blown predictive coding (gradient descent as MCMC [2]) help?</li>
</ul>

<p>I ask this question, because I can’t really find a process in human perception that’s equivalent to the discriminator, and we quite certainly do not tell the real from the fake while dreaming. Human perception largely relies on attention to sharpen things. But then again, humans aren’t that great at generating sharp images in the mind’s eye either. </p>

<p>On the other hand, when learning new motor actions, we do seem to use a discriminator (internal judgement and external critic). Also, while painting, we certainly use some internal judgement. Very interestingly, this “internal judgement” can be obtained solely through observation. For example, if I’ve seen a lot of van Gogh’s paintings, I’d be able to tell whether a painting looks like it’s done by him, even if I’ve never used a paintbrush in my life.</p>

<p>This “internal judgement”, which can be obtained through observation alone without generation, leads me to thinking: can the recognition model itself serve as the discriminator?</p>
