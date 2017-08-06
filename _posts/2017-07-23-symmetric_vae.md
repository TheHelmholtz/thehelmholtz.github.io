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
  <p>Notation: <script type="math/tex" id="MathJax-Element-1">Q(x)</script> is the real data distribution. <script type="math/tex" id="MathJax-Element-2">Q(z|x)</script> is the approximate posterior. <script type="math/tex" id="MathJax-Element-3">P(z)</script> is the prior over the latent code and <script type="math/tex" id="MathJax-Element-4">P(x|z)</script> is the likelihood of the generative model. Also, summations such as <script type="math/tex" id="MathJax-Element-5">\sum_{x,z}Q(x,z)</script> imply Monte Carlo integration, so gradient is not taken on these terms. <strong>I’ve used summation instead of expectation</strong> to make my LaTeX code readable. Also, all lower-case letters are vectors.</p>
</blockquote>

<p>The objective to be minimized in VAE [9] is:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-6">\begin{align*}
     &\sum_x Q(x) \sum_z Q(z|x)log\frac{Q(z|x)}{P(x,z)}
\\=&\sum_x Q(x)[KLD( Q(z|x) || P(z|x) ) + log\frac{1}{P(x)}]
\end{align*}</script></p>

<p>VAE is asymmetric in the sense that sampling is performed in only one direction (from <script type="math/tex" id="MathJax-Element-7">Q</script>). In comparison, both Boltzmann Machine [10] and Helmholtz Machine [11] perform sampling from both <script type="math/tex" id="MathJax-Element-8">Q</script> (real data) and <script type="math/tex" id="MathJax-Element-9">P</script> (fantasy data). The disadvantage of sampling from only <script type="math/tex" id="MathJax-Element-10">Q</script> is that there will be regions in <script type="math/tex" id="MathJax-Element-11">(x,z)</script> whose probability under <script type="math/tex" id="MathJax-Element-12">Q</script> is zero, yet whose probability under <script type="math/tex" id="MathJax-Element-13">P</script> is greater than zero. This means that when we sample from <script type="math/tex" id="MathJax-Element-14">P(x)</script>, we will get samples that are impossible under <script type="math/tex" id="MathJax-Element-15">Q(x)</script>. In the image domain, this means that when we sample from the model we will get images that do not look real.</p>

<p><img src="https://lh3.googleusercontent.com/-G6nSM2ag-oU/WVe04A09biI/AAAAAAAADgw/SP8_NeS-vRE6Ckyf-hIGn-B4YBxHfVDggCLcBGAs/s0/Q_and_P.png" alt="enter image description here" title="Q_and_P.png"></p>

<p>The diagram above illustrate this possibility. Because training is performed by sampling exclusively from <script type="math/tex" id="MathJax-Element-16">Q</script>, the regions that are not covered by <script type="math/tex" id="MathJax-Element-17">Q</script> may misbehave under <script type="math/tex" id="MathJax-Element-18">P</script>.</p>

<blockquote>
  <p>An intuitive interpretation: consider the case where a teacher is teaching a student to solve some problems. The teacher has limited time, so he only explains a subset of all possible problems. Let’s call the problems that the teacher has shown to the student the “taught problems”, and the rest “untaught problems”. </p>
  
  <p>If the student blindly accepts whatever the teacher teaches, he will not be able to handle untaught problems that are different from the taught ones. However, if the student is very curious, and frequently asks questions, and the teacher in turn <em>gives answers</em>, then we can reasonably expect this curious student to be much better at solving the untaught problems.</p>
  
  <p>When sampling from <script type="math/tex" id="MathJax-Element-19">Q</script>, the teacher is teaching the “taught problems”. When sampling from <script type="math/tex" id="MathJax-Element-20">P</script>, the student is asking questions about “untaught problems”. As we will later see, the discriminator in GANs is the device that “answers” the student’s questions.</p>
</blockquote>



<h2 id="symmetric-vae">Symmetric VAE</h2>

<p>In the Symmetric VAE, the objective has two parts. The first part is the original VAE objective, with sampling performed from <script type="math/tex" id="MathJax-Element-21">Q(x,z)</script>. The second part is its symmetric counterpart, with sampling performed from <script type="math/tex" id="MathJax-Element-22">P(x,z)</script>. It is written below:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-23">\begin{align*}
&\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(x,z)}
\end{align*}</script></p>

<p>This objective is an upper bound on <script type="math/tex" id="MathJax-Element-24">-logP(x)</script> and <script type="math/tex" id="MathJax-Element-25">-logQ(z)</script> (plus two KLDs), which is exactly what we want. Unfortunately, we cannot optimize the above objective directly, because it involves <script type="math/tex" id="MathJax-Element-26">Q(x,z)=Q(x)Q(z|x)</script>, which can be sampled from but cannot be computed (recall <script type="math/tex" id="MathJax-Element-27">Q(x)</script> is the real data distribution). To optimize the above objective, we need to approximate <script type="math/tex" id="MathJax-Element-28">Q(x)</script>, and replace <script type="math/tex" id="MathJax-Element-29">Q(x)</script> with its approximator. In doing this, we will obtain the discriminator of GANs [12].</p>



<h3 id="approximating-qx">Approximating <script type="math/tex" id="MathJax-Element-590">Q(x)</script></h3>

<p>We can train a normalized probability <script type="math/tex" id="MathJax-Element-591">Q'(x)</script> to approximate <script type="math/tex" id="MathJax-Element-592">Q(x)</script> by minimizing the cross entropy: <br>
<script type="math/tex; mode=display" id="MathJax-Element-593">\begin{align*}
      &\sum_x Q(x)log\frac{1}{Q'(x)}
\end{align*}</script></p>

<p>When <script type="math/tex" id="MathJax-Element-594">Q'(x)</script> and <script type="math/tex" id="MathJax-Element-595">Q(x)</script> are identical, we can recover the original Symmetric VAE objective exactly.</p>

<p>One complication with the above approach is that training with <script type="math/tex" id="MathJax-Element-596">Q'(x)</script>, a normalized probability, is difficult. So instead we train with <script type="math/tex" id="MathJax-Element-597">Q''(x)</script>, an unnormalized probability of <script type="math/tex" id="MathJax-Element-598">Q'(x)</script>. As it turns out, in order to train <script type="math/tex" id="MathJax-Element-599">Q''(x)</script> to approximate <script type="math/tex" id="MathJax-Element-600">Q(x)</script>, we need to minimize the unnormalized cross entropy from <script type="math/tex" id="MathJax-Element-601">Q(x)</script> to <script type="math/tex" id="MathJax-Element-602">Q''(x)</script>, and at the same time we need to maximize the unnormalized cross entropy of <script type="math/tex" id="MathJax-Element-603">P(x)</script> to <script type="math/tex" id="MathJax-Element-604">Q''(x)</script>, as in:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-629">
\sum_x Q(x) log\frac{1}{Q''(x)} - \sum_x P(x) log\frac{1}{Q''(x)}
</script></p>

<p>This is the discriminator loss minimized in WGAN [1]. So <script type="math/tex" id="MathJax-Element-630">logQ''(x)</script> is the discriminator/critic. Please refer to Appendix A for the derivation of the discriminator loss.</p>

<blockquote>
  <p>A comment on symmetry: <script type="math/tex" id="MathJax-Element-631">P(z)</script> is the prior over the “latent domain”. It encourages latent samples to be sparse (or, to have small magnitude, in the case of Gaussian priors). Correspondingly, <script type="math/tex" id="MathJax-Element-632">Q(x)</script> is the prior over the “visible domain”, and it encourages visible samples to look “real”.</p>
  
  <p>Previously we talked about a teacher answering the student’s questions. <script type="math/tex" id="MathJax-Element-633">Q(x)</script>, or its approximation, <script type="math/tex" id="MathJax-Element-634">Q''(x)</script>, is this teacher. If we want the student to learn and improve its work, it will be much better if the teacher can provide “useful guidance” instead of just bashing on the student when he misbehaves. We will further explore this teaching interpretation later.</p>
</blockquote>

<h2 id="extended-helmholtz-machine">Extended Helmholtz Machine</h2>

<p>After some experiments, we found it extremely difficult to optimize a Symmetric VAE, possibly because of the opponency that exists in the wake phase and the sleep phase. Specifically, the wake phase objective minimizes <script type="math/tex" id="MathJax-Element-106">logQ(z|x)</script>, yet the sleep phase maximizes it. Also, the wake phase objective maximizes <script type="math/tex" id="MathJax-Element-107">logP(x|z)</script>, while the sleep phase minimizes it.</p>

<p>If we remove the opponency, we get the objective of the Extended Helmholtz Machine, which minimizes two cross-entropies:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-54">\begin{align*}
     &\sum_{x,z}Q(x,z)log\frac{1}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{1}{Q(x,z)}
\end{align*}</script></p>

<p>This objective is equivalent to the original Symmetric VAE objective, plus two entropy terms <script type="math/tex" id="MathJax-Element-55">\sum_{x,z} Q(x,z)log\frac{1}{Q(z|x)}</script> and <script type="math/tex" id="MathJax-Element-56">\sum_{x,z} P(x,z)log\frac{1}{P(x|z)}</script>.</p>



<h3 id="results-of-ehm">Results of EHM</h3>

<p>Here we provide some very early results of training the EHM. The images below are sampled from an EHM whose encoder, decoder and discriminator have structures similar to DCGAN [8]. This model is trained on 64x64 LSUN bedroom images. </p>

<p><img src="https://lh3.googleusercontent.com/-CALIt-h6MkQ/WX84FF62ruI/AAAAAAAADjU/clpwWghMihY4-Xz0SlqAvqdfaCDUA3xVQCLcBGAs/s800/ehm7_samples_01_014000.png" alt="enter image description here" title="ehm7_samples_01_014000.png"></p>

<h3 id="current-problems">Current Problems</h3>

<p>There are several problems we have not been able to solve with EHM yet.</p>

<p><strong>Problem 1</strong>: <em>Generated images have significant artifacts</em>. Some of the samples contain a kind of strange artifact (last column of row 4 and 5). This kind of artifact has been observed in other GAN methods before, and we are working on understanding such artifacts. For now, we give 3 hypotheses which are to be validated:</p>

<ol>
<li>Artifacts are introduced by wake-phase reconstruction, which forces the generator to enter strange locations (unlikely, since these artefacts have been observed in other GAN methods).</li>
<li>Artifacts are introduced by excessive gradient from the discriminator, which leaves training far from convergence. This can be suppressed if we apply weight clipping as in WGAN [1]. However, because generator receives gradient from several sources, once we weaken the gradient from the discriminator, it may no longer be trained towards the direction we want. Therefore, we cannot naively clip the discriminator weights. Several alternative solutions exist, one is to decay learning rate after every epoch by, say, half. Another is to put a hard, but carefully chosen, limit on gradient norm. Yet another is to penalize gradient norm [6].</li>
<li>Artifacts have something to do with batch normalization. Some authors have suggested replacing BN with weight normalization [7], and some suggested using layer normalization instead [6].</li>
</ol>

<p><strong>Problem 2</strong>: <em>Sleep-phase reconstruction cost remains high</em>. After training the EHM, the wake-phase reconstruction loss is quite low, yet the sleep-phase reconstruction cost is high. Apparently the moment when the decoder starts to generate sharp images, the sleep-phase reconstruction cost starts rising. The implication of this is not yet understood. Below we give several hypotheses:</p>

<ol>
<li>generator has very steep surface. Small change in latent code leads to large change in generated images</li>
<li>generator relies on discontinuity to generate sharp images, and inverting such a highly nonsmooth function is hard</li>
<li>most dimensions in the latent code are not utilized. Naturally we cannot reconstruct noise.</li>
</ol>

<p><strong>Problem 3</strong>: <em>Wake-phase reconstructed samples remain blurry</em>. Another problem with the current EHM is that, when we feed the encoder with a real image, and ask the generator to reconstruct it, the reconstruction error is relatively low, yet we get a rather blurry reconstruction. The image below demonstrates this. The first row is the input image, and second row the corresponding reconstruction.</p>

<p><img src="https://lh3.googleusercontent.com/-RAP0CIpG9VQ/WYZ1H1kyjOI/AAAAAAAADjs/vPlOBoUXLpIooW0DOSwWDkZsp4PfHIQpQCLcBGAs/s800/blur_recon.png" alt="enter image description here" title="blur_recon.png"></p>

<p>These reconstructed samples look nothing like samples directly obtained by sampling from <script type="math/tex" id="MathJax-Element-1535">P(z)</script> then mapping <script type="math/tex" id="MathJax-Element-1536">z</script> through the decoder. One possible explanation for this is that, we cannot rely on the posterior obtained by a single pass to do reconstruction. After several stages of spatial down-sampling, it is difficult to imagine how the reconstruction can stay sharp. Instead, we might need to infer the posterior by optimization [3] [2], then perform reconstruction using the inferred posterior.</p>

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
[12] Ian Goodfellow, et al., Generative Adversarial Networks</p>

<p><br><br><br><br><br><br></p>

<hr>

<p><br><br><br><br><br><br></p>



<h2 id="appendix">Appendix</h2>



<h3 id="a-discriminator-loss">A. Discriminator loss</h3>

<blockquote>
  <p>Note: summation implies Monte Carlo integration, as indicated at the beginning of this article.</p>
</blockquote>

<p>We train an unnormalized probability <script type="math/tex" id="MathJax-Element-59">Q''(x)</script> to match <script type="math/tex" id="MathJax-Element-60">Q(x)</script>. We begin by converting <script type="math/tex" id="MathJax-Element-61">Q''(x)</script> to a normalized form  <br>
 <script type="math/tex; mode=display" id="MathJax-Element-62">Q'(x)=\frac{1}{Z}Q''(x)</script></p>

<p>where <script type="math/tex" id="MathJax-Element-63">Z=\sum_x Q''(x)</script> is the partition function. It can be estimated using samples:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-1336">\begin{align*}
Z =& \sum_x Q''(x)
\\=& \sum_x P(x|z) \frac{Q''(x)}{P(x|z)} && \text{Importance sampling. Equality holds for any $z$}
\\=& \sum_z P(z) \sum_x P(x|z) \frac{Q''(x)}{P(x|z)} &&\text{Sum over $z$}
\\=& \sum_{x,z} P(x,z) \frac{Q''(x)}{P(x|z)}
\end{align*}</script></p>

<blockquote>
  <p>This estimator of <script type="math/tex" id="MathJax-Element-1337">Z</script> is in fact sampling from <script type="math/tex" id="MathJax-Element-1338">P(x)</script>, but does not require us to estimate <script type="math/tex" id="MathJax-Element-1339">P(x)</script> itself (as is required in the simpler estimator of <script type="math/tex" id="MathJax-Element-1340">Z=\sum_x P(x) \frac{Q''(x)}{P(x)}</script>). </p>
  
  <p>Also, strictly, we are not required to sample from <script type="math/tex" id="MathJax-Element-1341">P(x)</script>, however to reduce variance we have to sample from a distribution as similar to <script type="math/tex" id="MathJax-Element-1342">Q'(x)</script> as possible. We can’t use <script type="math/tex" id="MathJax-Element-1343">Q(x)</script> itself, so <script type="math/tex" id="MathJax-Element-1344">P(x)</script> is the best we have. Of course, if we can sample from <script type="math/tex" id="MathJax-Element-1345">Q'(x)</script> directly, that’ll be even better. But it’s not yet clear how this can be done, which is why we use importance sampling in the first place.</p>
</blockquote>

<p>Now, to approximate <script type="math/tex" id="MathJax-Element-1346">Q(x)</script>, we train <script type="math/tex" id="MathJax-Element-1347">Q''(x)</script> to minimize the cross entropy from <script type="math/tex" id="MathJax-Element-1348">Q(x)</script> to <script type="math/tex" id="MathJax-Element-1349">Q'(x)</script>:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-1121">\begin{align*}
     & \sum_x Q(x) log\frac{1}{Q'(x)}
\\=& \sum_x Q(x) [log\frac{1}{Q''(x)} + logZ]
\end{align*}</script></p>

<p>Now we will expand <script type="math/tex" id="MathJax-Element-1122">Z</script> to its estimator. Note that <script type="math/tex" id="MathJax-Element-1123">(x,z)</script> in <script type="math/tex" id="MathJax-Element-1124">Z</script> are sampled from <script type="math/tex" id="MathJax-Element-1125">P</script>, and is completely independent from the <script type="math/tex" id="MathJax-Element-1126">x</script> samples taken from <script type="math/tex" id="MathJax-Element-1127">Q</script>. To reflect this independence, we take out <script type="math/tex" id="MathJax-Element-1128">logZ</script> as an independent term. The above becomes: <br>
<script type="math/tex; mode=display" id="MathJax-Element-1129">\begin{align*}
     &\sum_v Q(x) log\frac{1}{Q''(x)} + logZ
\\=&\sum_v Q(x) log\frac{1}{Q''(x)} + log\sum_{x,z}P(x,z)\frac{Q''(x)}{P(x|z)} 
\\=&\sum_v Q(x)  log\frac{1}{Q''(x)}   - log\sum_{x,z}P(x,z)\frac{P(x|z)}{Q''(x)}
\\\le &\sum_v Q(x) log\frac{1}{Q''(x)}  - \sum_{x,z}P(x,z) log\frac{P(x|z)}{Q''(x)}
\end{align*}</script></p>

<p>Because we only minimize the above w.r.t. parameters of <script type="math/tex" id="MathJax-Element-1130">Q''(x)</script>, it is equivalent to minimizing:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-84">
\sum_v Q(v) log\frac{1}{Q''(v)} - \sum_x P(x) log\frac{1}{Q''(x)}
</script></p>

<p>This is the discriminator loss minimized by WGAN [1].</p>



<h3 id="b-generator-gradient">B. Generator Gradient</h3>

<p>For the sake of demonstrating VAE’s connection with GANs, we can also obtain a gradient term similar to the generator gradient in GANs. Since GANs do not have the inference network <script type="math/tex" id="MathJax-Element-85">Q(z|x)</script>, we remove all components of <script type="math/tex" id="MathJax-Element-86">Q(z|x)</script>, and replace <script type="math/tex" id="MathJax-Element-87">Q(x)</script> with <script type="math/tex" id="MathJax-Element-88">Q'(x)</script>:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-89">\begin{align*}
     &\sum_z P(z) \sum_x P(x|z)log\frac{P(x|z)}{Q'(x)}
\end{align*}</script></p>

<p>We differentiate <script type="math/tex" id="MathJax-Element-90">log\frac{P(x|z)}{Q'(x)}</script> against the decoder parameter <script type="math/tex" id="MathJax-Element-91">\phi</script> (the summations are Monte Carlo integration):</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-97">\begin{align*}
     &\nabla_\phi log\frac{P(x|z)}{Q'(x)}
\\=&  \nabla_\phi log\frac{1}{Q'(x)} 
       - \nabla_\phi log\frac{1}{P(x|z)}
\end{align*}</script></p>

<p>Note that in GANs, <script type="math/tex" id="MathJax-Element-98">P(x|z)=1</script>, but it isn’t the case for VAEs, which is why we have the second term above. The first term in the above can be written as: <br>
<script type="math/tex; mode=display" id="MathJax-Element-99">\begin{align*}
     & \nabla_x log \frac{1}{Q'(x)} \frac{\partial x}{\partial \phi}
\\=&[\nabla_x log \frac{1}{Q''(x)} + \nabla_x logZ]  \frac{\partial x}{\partial \phi}
\\=&\nabla_x log \frac{1}{Q''(x)} \frac{\partial x}{\partial \phi}
     && \text{$Z$ does not depend on $x$}
\end{align*}</script></p>

<p>Which is the generator gradient. The above also suggests that the unnormalized probability <script type="math/tex" id="MathJax-Element-100">Q''(x)</script> can be used directly to replace <script type="math/tex" id="MathJax-Element-101">Q(x)</script> in optimization.</p>
