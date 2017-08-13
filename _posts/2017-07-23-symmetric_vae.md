---
layout: post
title:  "Symmetric variational loss and GAN"
date:   2017-08-05 10:00:00 +0800
permalink: /blog/symmetric_vae
comments: true
categories: jekyll update
---



<p>In this post, we’ll look into a kind of variational autoencoder that tries to reconstruct both the input and the latent code. Along the way we’ll show how to derive GAN’s discriminator from such a variational loss. We’ll start with the fact that VAE’s sampling is asymmetric, and why this asymmetry might give us problems.</p>



<h2 id="vaes-asymmetry">VAE’s asymmetry</h2>

<blockquote>
  <p>Notation: <script type="math/tex" id="MathJax-Element-14778">Q(x)</script> is the real data distribution. <script type="math/tex" id="MathJax-Element-14779">Q(z|x)</script> is the approximate posterior. <script type="math/tex" id="MathJax-Element-14780">P(z)</script> is the prior over the latent code and <script type="math/tex" id="MathJax-Element-14781">P(x|z)</script> is the likelihood of the generative model. Also, summations such as <script type="math/tex" id="MathJax-Element-14782">\sum_{x,z}Q(x,z)</script> imply Monte Carlo integration, so gradient is not taken on these terms. <strong>I’ve used summation instead of expectation</strong> to make my LaTeX code readable. Also, all lower-case letters are vectors.</p>
</blockquote>

<p>The objective to be minimized in VAE [9] is:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-14783">\begin{align*}
     &\sum_x Q(x) \sum_z Q(z|x)log\frac{Q(z|x)}{P(x,z)}
\\=&\sum_x Q(x)[KLD( Q(z|x) || P(z|x) ) + log\frac{1}{P(x)}]
\end{align*}</script></p>

<p>VAE is asymmetric in the sense that sampling is performed in only one direction (from <script type="math/tex" id="MathJax-Element-14784">Q</script>). In comparison, both Boltzmann Machine [10] and Helmholtz Machine [11] perform sampling from both <script type="math/tex" id="MathJax-Element-14785">Q</script> (real data) and <script type="math/tex" id="MathJax-Element-14786">P</script> (fantasy data). The disadvantage of sampling from only <script type="math/tex" id="MathJax-Element-14787">Q</script> is that there will be regions in <script type="math/tex" id="MathJax-Element-14788">(x,z)</script> whose probability under <script type="math/tex" id="MathJax-Element-14789">Q</script> is zero, yet whose probability under <script type="math/tex" id="MathJax-Element-14790">P</script> is greater than zero. This means that when we sample from <script type="math/tex" id="MathJax-Element-14791">P(x)</script>, we will get samples that are impossible under <script type="math/tex" id="MathJax-Element-14792">Q(x)</script>. In the image domain, this means that when we sample from the model we will get images that do not look real.</p>

<p><img src="https://lh3.googleusercontent.com/-G6nSM2ag-oU/WVe04A09biI/AAAAAAAADgw/SP8_NeS-vRE6Ckyf-hIGn-B4YBxHfVDggCLcBGAs/s0/Q_and_P.png" alt="enter image description here" title="Q_and_P.png"></p>

<p>The diagram above illustrate this possibility. Because training is performed by sampling exclusively from <script type="math/tex" id="MathJax-Element-14793">Q</script>, the regions that are not covered by <script type="math/tex" id="MathJax-Element-14794">Q</script> may misbehave under <script type="math/tex" id="MathJax-Element-14795">P</script>.</p>

<blockquote>
  <p>An intuitive interpretation: consider the case where a teacher is teaching a student to solve some problems. The teacher has limited time, so he only explains a subset of all possible problems. Let’s call the problems that the teacher has shown to the student the “taught problems”, and the rest “untaught problems”. </p>
  
  <p>If the student blindly accepts whatever the teacher teaches, he will not be able to handle untaught problems that are different from the taught ones. However, if the student is very curious, and frequently asks questions, and the teacher in turn <em>gives answers</em>, then we can reasonably expect this curious student to be much better at solving the untaught problems.</p>
  
  <p>When sampling from <script type="math/tex" id="MathJax-Element-14796">Q</script>, the teacher is teaching the “taught problems”. When sampling from <script type="math/tex" id="MathJax-Element-14797">P</script>, the student is asking questions about “untaught problems”. As we will later see, the discriminator in GANs is the device that “answers” the student’s questions.</p>
</blockquote>



<h2 id="symmetric-vae">Symmetric VAE</h2>

<p>In the Symmetric VAE, the objective has two parts. The first part is the original VAE objective, with sampling performed from <script type="math/tex" id="MathJax-Element-14798">Q(x,z)</script>. The second part is its symmetric counterpart, with sampling performed from <script type="math/tex" id="MathJax-Element-14799">P(x,z)</script>. It is written below:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-14800">\begin{align*}
&\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(x,z)}
\end{align*}</script></p>

<p>This objective is an upper bound on <script type="math/tex" id="MathJax-Element-14801">-logP(x)</script> and <script type="math/tex" id="MathJax-Element-14802">-logQ(z)</script> (plus two KLDs), which is exactly what we want. Unfortunately, we cannot optimize the above objective directly, because it involves <script type="math/tex" id="MathJax-Element-14803">Q(x,z)=Q(x)Q(z|x)</script>, which can be sampled from but cannot be computed (recall <script type="math/tex" id="MathJax-Element-14804">Q(x)</script> is the real data distribution). To optimize the above objective, we need to approximate <script type="math/tex" id="MathJax-Element-14805">Q(x)</script>, and replace <script type="math/tex" id="MathJax-Element-14806">Q(x)</script> with its approximator. In doing this, we will obtain the discriminator of GANs [12].</p>



<h3 id="approximating-qx">Approximating <script type="math/tex" id="MathJax-Element-14807">Q(x)</script></h3>

<blockquote>
  <p>A comment on symmetry: <script type="math/tex" id="MathJax-Element-14808">P(z)</script> is the prior over the “latent domain”. It encourages latent samples to be sparse (or, to have small magnitude, in the case of Gaussian priors). Correspondingly, <script type="math/tex" id="MathJax-Element-14809">Q(x)</script> is the prior over the “visible domain”, and it encourages visible samples to look “real”.</p>
</blockquote>

<p>We can train a normalized probability <script type="math/tex" id="MathJax-Element-14810">Q'(x)</script> to approximate <script type="math/tex" id="MathJax-Element-14811">Q(x)</script> by minimizing the cross entropy: <br>
<script type="math/tex; mode=display" id="MathJax-Element-14812">\begin{align*}
      &\sum_x Q(x)log\frac{1}{Q'(x)}
\end{align*}</script></p>

<p>When <script type="math/tex" id="MathJax-Element-14813">Q'(x)</script> and <script type="math/tex" id="MathJax-Element-14814">Q(x)</script> are identical, we can recover the original Symmetric VAE objective exactly.</p>

<p>One complication with the above approach is that training with <script type="math/tex" id="MathJax-Element-14815">Q'(x)</script>, a normalized probability, is difficult. So instead we train with <script type="math/tex" id="MathJax-Element-14816">Q''(x)</script>, an unnormalized probability of <script type="math/tex" id="MathJax-Element-14817">Q'(x)</script>. As it turns out, in order to train <script type="math/tex" id="MathJax-Element-14818">Q''(x)</script> to approximate <script type="math/tex" id="MathJax-Element-14819">Q(x)</script>, we need to minimize the unnormalized cross entropy from <script type="math/tex" id="MathJax-Element-14820">Q(x)</script> to <script type="math/tex" id="MathJax-Element-14821">Q''(x)</script>, and at the same time we need to maximize the unnormalized cross entropy of <script type="math/tex" id="MathJax-Element-14822">P(x)</script> to <script type="math/tex" id="MathJax-Element-14823">Q''(x)</script>, as in:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-16363">
\sum_x Q(x) log\frac{1}{Q''(x)} - \sum_x P(x) log\frac{1}{Q''(x)}
</script></p>

<p>This is the discriminator loss minimized in WGAN [1]. So <script type="math/tex" id="MathJax-Element-16364">logQ''(x)</script> is the discriminator/critic. Please refer to Appendix A for how this loss is derived from the perspective of approximating <script type="math/tex" id="MathJax-Element-16365">Q(x)</script>.</p>

<blockquote>
  <p>Previously we talked about a teacher answering the student’s questions. <script type="math/tex" id="MathJax-Element-16366">Q(x)</script>, or its approximation, <script type="math/tex" id="MathJax-Element-16367">Q''(x)</script>, is this teacher. If we want the student to learn and improve its work, it will be much better if the teacher can provide “useful guidance” instead of just bashing on the student when he misbehaves. We will further explore this teaching interpretation later.</p>
</blockquote>



<h3 id="approximating-pz">Approximating <script type="math/tex" id="MathJax-Element-16401">P(z)</script></h3>

<p>Even though <script type="math/tex" id="MathJax-Element-16402">P(z)</script> is explicitly available, we still approximate it with <script type="math/tex" id="MathJax-Element-16403">logP''(z)</script> by minimizing:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-16582">
- \sum_z Q(z) log\frac{1}{P''(z)} + \sum_z P(z) log\frac{1}{P''(z)}
</script></p>

<p>We then use <script type="math/tex" id="MathJax-Element-16583">logP''(z)</script> to replace <script type="math/tex" id="MathJax-Element-16584">logP(z)</script> in the objective. </p>

<p>We have found out that when we simply regularize <script type="math/tex" id="MathJax-Element-16585">z</script> using <script type="math/tex" id="MathJax-Element-16586">P(z)</script>, the resulting latent distribution lies in a very small subspace covered by <script type="math/tex" id="MathJax-Element-16587">P(z)</script>. To correct this asymmetry, we use the approximator to drive it away from that small subspace. This has significantly improved the experiment result.</p>

<h2 id="extended-helmholtz-machine">Extended Helmholtz Machine</h2>

<p>After some experiments, we found it extremely difficult to optimize a Symmetric VAE, possibly because of the opponency that exists in the wake phase and the sleep phase. Specifically, the wake phase objective minimizes <script type="math/tex" id="MathJax-Element-16426">logQ(z|x)</script>, yet the sleep phase maximizes it. Also, the wake phase objective maximizes <script type="math/tex" id="MathJax-Element-16427">logP(x|z)</script>, while the sleep phase minimizes it.</p>

<p>If we remove the opponency, we get the objective of the Extended Helmholtz Machine, which minimizes two cross-entropies:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-16428">\begin{align*}
     &\sum_{x,z}Q(x,z)log\frac{1}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{1}{Q(x,z)}
\end{align*}</script></p>

<p>This objective is equivalent to the original Symmetric VAE objective, plus two entropy terms <script type="math/tex" id="MathJax-Element-16429">\sum_{x,z} Q(x,z)log\frac{1}{Q(z|x)}</script> and <script type="math/tex" id="MathJax-Element-16430">\sum_{x,z} P(x,z)log\frac{1}{P(x|z)}</script>.</p>



<h3 id="results-of-ehm">Results of EHM</h3>

<p>Here we provide some very early results of training the EHM. The images below are sampled from an EHM whose encoder, decoder and discriminator have structures similar to DCGAN [8]. This model is trained on 64x64 LSUN bedroom images. </p>

<p><img src="https://lh3.googleusercontent.com/-CALIt-h6MkQ/WX84FF62ruI/AAAAAAAADjU/clpwWghMihY4-Xz0SlqAvqdfaCDUA3xVQCLcBGAs/s800/ehm7_samples_01_014000.png" alt="enter image description here" title="ehm7_samples_01_014000.png"></p>



<h3 id="current-problems">Current Problems</h3>

<p>There are several problems we have not been able to solve with EHM yet.</p>

<p><strong>Problem 1</strong>: <em>Generated images have significant artifacts</em>. Some of the samples contain a kind of strange artifact (last column of row 4 and 5). This kind of artifact has been observed in other GAN methods before, and we are working on understanding such artifacts. Previously we had quite a few hypotheses about the artefacts, now we have narrowed it down to just one: large variance in gradients from the discriminator.</p>

<p>Basically, at the beginning of training, the generated distribution <script type="math/tex" id="MathJax-Element-16940">P(x)</script> and <script type="math/tex" id="MathJax-Element-16941">Q(x)</script> have no overlap in support, as a result, the discriminator is able to perfectly discriminate. It does so by blowing up the magnitude of <script type="math/tex" id="MathJax-Element-16942">logQ''(x)</script> to very big values, using very big weights in the discriminator. This creates very steep gradients in the form of <script type="math/tex" id="MathJax-Element-16943">\nabla_x logQ''(x)</script>. Such large gradient variance leads to problematic learning.</p>

<p>As proposed in WGAN [1], we can use weight clipping to restrict this kind of variance. Alternatively, we can, as in improved WGAN [6], regularize the gradient norm towards a target. However, because generator receives gradient from several sources, once we weaken the gradient from the discriminator using heuristics, it may no longer be trained towards the direction we want. Therefore, we cannot naively tune the gradients. However, since we now recognize that the necessary mechanism is variance reduction, we resort to SVRG [14], but we have yet to conduct experiments with it.</p>

<blockquote>
  <p>Note: the discriminator, in RL’s terminology, provides a <em>reward</em>, and the generator is trained through <em>policy gradient</em>. However, traditional policy gradient methods use the REINFORCE estimator, whereas in GANs, the pathwise gradient term is taken instead. Just like in conventional policy gradient methods, variance reduction is critical. In GANs, we cannot use naive baseline methods as it would bias the gradient. Fortunately, the baseline provided by SVRG has small bias, which is why we prefer it. Also, we’ve developed a variant of SVRG in <a href="/blog/ctc_variance">this post</a>, which we hope will be more easily adapted to GANs (no need to keep a stale copy of the network), but we have yet to run experiment with it.</p>
</blockquote>

<p><strong>Problem 2</strong>: <em>Sleep-phase reconstruction cost remains high</em>. <strong>Update</strong>: this problem has been fixed by replacing <script type="math/tex" id="MathJax-Element-16944">P(z)</script> with its adversarial approximator.</p>

<p><strong>Problem 3</strong>: <em>Wake-phase reconstructed samples remain blurry</em>. Another problem with the current EHM is that, when we feed the encoder with a real image, and ask the generator to reconstruct it, the reconstruction error is relatively low, yet we get a rather blurry reconstruction. The image below demonstrates this. The first row is the input image, and second row the corresponding reconstruction.</p>

<p><img src="https://lh3.googleusercontent.com/-RAP0CIpG9VQ/WYZ1H1kyjOI/AAAAAAAADjs/vPlOBoUXLpIooW0DOSwWDkZsp4PfHIQpQCLcBGAs/s800/blur_recon.png" alt="enter image description here" title="blur_recon.png"></p>

<p>These reconstructed samples look nothing like samples directly obtained by sampling from <script type="math/tex" id="MathJax-Element-16945">P(z)</script> then mapping <script type="math/tex" id="MathJax-Element-16946">z</script> through the decoder. <strong>Update</strong>: The reason, we have now figured out, is that <script type="math/tex" id="MathJax-Element-16947">Q(z)</script> only occupies a very small subspace of <script type="math/tex" id="MathJax-Element-16948">P(z)</script>. To correct this, we have recently swapped <script type="math/tex" id="MathJax-Element-16949">P(z)</script> with its adversarial approximator. This has led to reconstructions that are sharp but that produce significantly higher pixel-level inaccuracy, which is exactly what we expect.</p>

<p>TBD: show images of reconstruction with adversarial approximator as <script type="math/tex" id="MathJax-Element-16950">P(z)</script>.</p>

<h2 id="conclusion">Conclusion</h2>

<ol>
<li>We argue that probability models should be trained with symmetric sampling</li>
<li>We showed that GAN can be derived from the sleep-phase of a Symmetric VAE</li>
<li>We showed that the Extended Helmholtz Machine can be trained, but there are still many problems.</li>
</ol>



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
[12] Ian Goodfellow, et al., Generative Adversarial Networks <br>
[13] Radford Neal, MCMC using Hamiltonian dynamics <br>
[14] Rie Johnson, et al., Accelerating Stochastic Gradient Descent using Predictive Variance Reduction</p>

<p><br><br><br><br><br><br></p>

<hr>

<p><br><br><br><br><br><br></p>



<h2 id="appendix">Appendix</h2>



<h3 id="a-discriminator-loss">A. Discriminator loss</h3>

<blockquote>
  <p>Note: summation implies Monte Carlo integration, as indicated at the beginning of this article.</p>
</blockquote>

<p>We train an unnormalized probability <script type="math/tex" id="MathJax-Element-16442">Q''(x)</script> to match <script type="math/tex" id="MathJax-Element-16443">Q(x)</script>. We begin by converting <script type="math/tex" id="MathJax-Element-16444">Q''(x)</script> to a normalized form  <br>
 <script type="math/tex; mode=display" id="MathJax-Element-16445">Q'(x)=\frac{1}{Z}Q''(x)</script></p>

<p>where <script type="math/tex" id="MathJax-Element-16446">Z=\sum_x Q''(x)</script> is the partition function. It can be estimated using samples:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-14841">\begin{align*}
Z =& \sum_x Q''(x)
\\=& \sum_x P(x|z) \frac{Q''(x)}{P(x|z)} && \text{Importance sampling. Equality holds for any $z$}
\\=& \sum_z P(z) \sum_x P(x|z) \frac{Q''(x)}{P(x|z)} &&\text{Sum over $z$}
\\=& \sum_{x,z} P(x,z) \frac{Q''(x)}{P(x|z)}
\end{align*}</script></p>

<blockquote>
  <p>This estimator of <script type="math/tex" id="MathJax-Element-14842">Z</script> is in fact sampling from <script type="math/tex" id="MathJax-Element-14843">P(x)</script>, but does not require us to estimate <script type="math/tex" id="MathJax-Element-14844">P(x)</script> itself (as is required in the simpler estimator of <script type="math/tex" id="MathJax-Element-14845">Z=\sum_x P(x) \frac{Q''(x)}{P(x)}</script>). </p>
  
  <p>Also, strictly, we are not required to sample from <script type="math/tex" id="MathJax-Element-14846">P(x)</script>, however to reduce variance we have to sample from a distribution as similar to <script type="math/tex" id="MathJax-Element-14847">Q'(x)</script> as possible. We can’t use <script type="math/tex" id="MathJax-Element-14848">Q(x)</script> itself, so <script type="math/tex" id="MathJax-Element-14849">P(x)</script> is the best we have. </p>
  
  <p>Of course, if we can sample from <script type="math/tex" id="MathJax-Element-14850">Q'(x)</script> directly, that’ll be even better. But this usually isn’t straightforward, which is why we use importance sampling in the first place. In order to sample from <script type="math/tex" id="MathJax-Element-14851">Q'(x)</script>, we might have to use some variants of MCMC (Hamiltonian [13] or Langevin [2] dynamics). If we could do this, it means that we’ll be able to use the discriminator as the generator. Sounds like a pretty fun project, but we’ll leave that for another post.</p>
</blockquote>

<p>Now, to approximate <script type="math/tex" id="MathJax-Element-14852">Q(x)</script>, we train <script type="math/tex" id="MathJax-Element-14853">Q''(x)</script> to minimize the cross entropy from <script type="math/tex" id="MathJax-Element-14854">Q(x)</script> to <script type="math/tex" id="MathJax-Element-14855">Q'(x)</script>:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-14856">\begin{align*}
     & \sum_x Q(x) log\frac{1}{Q'(x)}
\\=& \sum_x Q(x) [log\frac{1}{Q''(x)} + logZ]
\end{align*}</script></p>

<p>Now we will expand <script type="math/tex" id="MathJax-Element-14857">Z</script> to its estimator. Note that <script type="math/tex" id="MathJax-Element-14858">(x,z)</script> in <script type="math/tex" id="MathJax-Element-14859">Z</script> are sampled from <script type="math/tex" id="MathJax-Element-14860">P</script>, and is completely independent from the <script type="math/tex" id="MathJax-Element-14861">x</script> samples taken from <script type="math/tex" id="MathJax-Element-14862">Q</script>. To reflect this independence, we take out <script type="math/tex" id="MathJax-Element-14863">logZ</script> as an independent term. The above becomes: <br>
<script type="math/tex; mode=display" id="MathJax-Element-14864">\begin{align*}
     &\sum_x Q(x) log\frac{1}{Q''(x)} + logZ
\\=&\sum_x Q(x) log\frac{1}{Q''(x)} + log\sum_{x,z}P(x,z)\frac{Q''(x)}{P(x|z)} 
\\=&\sum_x Q(x)  log\frac{1}{Q''(x)}   - log\sum_{x,z}P(x,z)\frac{P(x|z)}{Q''(x)}
\\\le &\sum_x Q(x) log\frac{1}{Q''(x)}  - \sum_{x,z}P(x,z) log\frac{P(x|z)}{Q''(x)}
\end{align*}</script></p>

<p>Because we only minimize the above w.r.t. parameters of <script type="math/tex" id="MathJax-Element-14865">Q''(x)</script>, it is equivalent to minimizing:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-14866">
\sum_x Q(x) log\frac{1}{Q''(x)} - \sum_x P(x) log\frac{1}{Q''(x)}
</script></p>

<p>This is the discriminator loss minimized by WGAN [1].</p>



<h3 id="b-generator-gradient">B. Generator Gradient</h3>

<p>For the sake of demonstrating VAE’s connection with GANs, we can also obtain a gradient term similar to the generator gradient in GANs. Since GANs do not have the inference network <script type="math/tex" id="MathJax-Element-14867">Q(z|x)</script>, we remove all components of <script type="math/tex" id="MathJax-Element-14868">Q(z|x)</script>, and replace <script type="math/tex" id="MathJax-Element-14869">Q(x)</script> with <script type="math/tex" id="MathJax-Element-14870">Q'(x)</script>:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-14871">\begin{align*}
     &\sum_z P(z) \sum_x P(x|z)log\frac{P(x|z)}{Q'(x)}
\end{align*}</script></p>

<p>We differentiate <script type="math/tex" id="MathJax-Element-14872">log\frac{P(x|z)}{Q'(x)}</script> against the decoder parameter <script type="math/tex" id="MathJax-Element-14873">\phi</script> (the summations are Monte Carlo integration):</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-14874">\begin{align*}
     &\nabla_\phi log\frac{P(x|z)}{Q'(x)}
\\=&  \nabla_\phi log\frac{1}{Q'(x)} 
       - \nabla_\phi log\frac{1}{P(x|z)}
\end{align*}</script></p>

<p>Note that in GANs, <script type="math/tex" id="MathJax-Element-14875">P(x|z)=1</script>, but it isn’t the case for VAEs, which is why we have the second term above. The first term in the above can be written as: <br>
<script type="math/tex; mode=display" id="MathJax-Element-14876">\begin{align*}
     & \nabla_x log \frac{1}{Q'(x)} \frac{\partial x}{\partial \phi}
\\=&[\nabla_x log \frac{1}{Q''(x)} + \nabla_x logZ]  \frac{\partial x}{\partial \phi}
\\=&\nabla_x log \frac{1}{Q''(x)} \frac{\partial x}{\partial \phi}
     && \text{$Z$ does not depend on $x$}
\end{align*}</script></p>

<p>Which is the generator gradient. The above also suggests that the unnormalized probability <script type="math/tex" id="MathJax-Element-14877">Q''(x)</script> can be used directly to replace <script type="math/tex" id="MathJax-Element-14878">Q(x)</script> in optimization.</p>
