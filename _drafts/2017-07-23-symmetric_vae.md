---
layout: post
title:  "Symmetric VAE derives GAN"
date:   2017-07-23 10:00:00 +0800
permalink: /blog/symmetric_vae
comments: true
categories: jekyll update
---


In this post, we discuss the asymmetry of traditional Variational Autoencoders, and why it might be better to use a
symmetric cost instead. We then introduce the the Symmetric VAE, a type of VAE with symmetric sampling, and demonstrate
how GAN can be derived from it.

<h1 id="summary-of-results">Summary of results</h1>

<blockquote>
  <p>Notation: <script type="math/tex" id="MathJax-Element-1423">Q(x)</script> is the data distribution. <script type="math/tex" id="MathJax-Element-1424">Q(z|x)</script> is the approximate posterior. <script type="math/tex" id="MathJax-Element-1425">P</script> is the generative model. Also, <script type="math/tex" id="MathJax-Element-1426">\sum_{x,z}Q(x,z)</script> and several similar forms in this work imply Monte Carlo integration, so gradient is not taken on these terms. I’ve used summation instead of expectation to make my LaTeX code readable. Also, all lower-case letters are vectors.</p>
</blockquote>

<p>The objective of the Symmetric VAE (the MDL objective): <br>
<script type="math/tex; mode=display" id="MathJax-Element-1427">
\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(x,z)}
</script></p>

<p>The above cannot be directly computed. When approximating this objective, we approximate <script type="math/tex" id="MathJax-Element-1428">Q(x)</script> with an unnormalized distribution <script type="math/tex" id="MathJax-Element-1429">Q''(x)</script> by minimizing a discriminator loss:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-1366">
E_{x \sim Q(x)}log\frac{1}{Q''(x)} - E_{x \sim P(x)}log\frac{1}{Q''(x)}
</script></p>

<h1 id="vaes-asymmetry">VAE’s asymmetry</h1>

<p>The objective of VAE is:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-894">\begin{align*}
     &\sum_x Q(x) \sum_z Q(z|x)log\frac{Q(z|x)}{P(x,z)}
\\=&\sum_x Q(x)[KLD( Q(z|x) || P(z|x) ) + log\frac{1}{P(x)}]
\end{align*}</script></p>

<p>Note that <script type="math/tex" id="MathJax-Element-895">Q(x)</script> is the data distribution, <script type="math/tex" id="MathJax-Element-896">Q(z|x)</script> the approximate posterior, and <script type="math/tex" id="MathJax-Element-897">P</script> the generative model.</p>

<p>VAE is asymmetric in the sense that sampling is performed in only one direction (from <script type="math/tex" id="MathJax-Element-898">Q</script>). In comparison, both Boltzmann Machine and Helmholtz Machine perform sampling from both <script type="math/tex" id="MathJax-Element-899">Q</script> (real data) and <script type="math/tex" id="MathJax-Element-900">P</script> (fantasy data). The disadvantage of sampling from only <script type="math/tex" id="MathJax-Element-901">Q</script> is that there will be regions in <script type="math/tex" id="MathJax-Element-902">(x,z)</script> whose probability under <script type="math/tex" id="MathJax-Element-903">Q</script> is zero, yet whose probability under <script type="math/tex" id="MathJax-Element-904">P</script> is greater than zero.</p>

<p><img src="https://lh3.googleusercontent.com/-G6nSM2ag-oU/WVe04A09biI/AAAAAAAADgw/SP8_NeS-vRE6Ckyf-hIGn-B4YBxHfVDggCLcBGAs/s0/Q_and_P.png" alt="enter image description here" title="Q_and_P.png"></p>

<p>The image above illustrate this possibility. Because training is performed by sampling exclusively from <script type="math/tex" id="MathJax-Element-905">Q</script>, the regions that are not covered by <script type="math/tex" id="MathJax-Element-906">Q</script> may misbehave under <script type="math/tex" id="MathJax-Element-907">P</script>.</p>

<blockquote>
  <p>An intuitive interpretation: consider the case where a teacher is teaching a student to solve some problems. The teacher has limited time, so he only explains a subset of all possible problems. Let’s call the problems that the teacher has shown to the student the “taught problems”, and the rest “untaught problems”. </p>
  
  <p>If the student blindly accepts whatever the teacher teaches, he will not be able to handle untaught problems that are different from the taught ones. However, if the student is very curious, and frequently asks questions, and the teacher in turn gives answers, then we can reasonably expect this curious student to be much better at solving the untaught problems.</p>
  
  <p>When sampling from <script type="math/tex" id="MathJax-Element-908">Q</script>, the teacher is teaching the “taught problems”. When sampling from <script type="math/tex" id="MathJax-Element-909">P</script>, the student is asking questions about “untaught problems”.</p>
</blockquote>

<p>In the next part, we briefly review the Boltzmann Machine and the Helmholtz Machine, both of which optimize a symmetric objective.</p>



<h1 id="symmetric-objectives">Symmetric objectives</h1>



<h2 id="boltzmann-machine">Boltzmann Machine</h2>

<p>Boltzmann Machine maximizes the following objective:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-910">
          \frac{\sum_h U(v,h)}{\sum_{x,z}U(x,z)}
       = \frac{\sum_h \frac{1}{Z}U(v,h)}{\sum_{x,z}\frac{1}{Z}U(x,z)}
       = \frac{\sum_h P(v,h)}{\sum_{x,z}P(x,z)}
       = \frac{P(v)}{1}
</script></p>

<ul>
<li>Positive phase: maximize probability of observed data</li>
<li>Negative phase: minimize probability of fantasy data</li>
<li>Notation:  <br>
<ul><li><script type="math/tex" id="MathJax-Element-911">U</script> is unnormalized probability</li>
<li><script type="math/tex" id="MathJax-Element-912">v</script> is observed data, <script type="math/tex" id="MathJax-Element-913">h</script> is the corresponding latent state sample</li>
<li><script type="math/tex" id="MathJax-Element-914">x</script> and <script type="math/tex" id="MathJax-Element-915">z</script> are visible and latent samples of the fantasy distribution</li></ul></li>
</ul>

<p>The symmetry is obvious in the positive phase and the negative phase.</p>



<h2 id="helmholtz-machine">Helmholtz Machine</h2>

<p>The Wake-Sleep Algorithm was used to train the first Helmholtz Machine, which has a recognition model <script type="math/tex" id="MathJax-Element-916">Q</script> and a generative model <script type="math/tex" id="MathJax-Element-917">P</script>. It is very similar to modern VAEs, but optimizes a different objective:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-1289">\begin{align*}
&\sum_{x,z}Q(x,z)log\frac{1}{P(x|z)} + \sum_{x,z}P(x,z)log\frac{1}{Q(z|x)}
\\=&\sum_{x,z}Q(x,z)log\frac{Q(x|z)}{P(x|z)Q(x|z)} + \sum_{x,z}P(x,z)log\frac{P(z|x)}{Q(z|x)P(z|x)}
\\=&\sum_z Q(z)[KLD( Q(x|z) || P(x|z) ) + H(Q(x|z))] + \\
     &\sum_x P(x)[KLD( P(z|x) || Q(z|x) ) + H(P(z|x))]
\end{align*}</script></p>

<ul>
<li>Wake phase: bring <script type="math/tex" id="MathJax-Element-1290">P</script> to <script type="math/tex" id="MathJax-Element-1291">Q</script></li>
<li>Sleep phase: bring <script type="math/tex" id="MathJax-Element-1292">Q</script> to <script type="math/tex" id="MathJax-Element-1293">P</script></li>
</ul>

<p>Again, the symmetry is obvious in the wake phase and the sleep phase.</p>

<p>There are two problems associated with traditional Helmholtz Machines</p>

<ol>
<li>It does not optimize the proper Minimum Description Length objective</li>
<li>It cannot handle multimodal posterior</li>
</ol>

<p>In this work, we tackle the first problem. We begin by giving the Helmholtz Machine a proper Minimum Description Length objective.</p>

<h1 id="symmetric-vae">Symmetric VAE</h1>

<blockquote>
  <p>Note: Symmetric VAE = Extended Helmholtz Machine.</p>
</blockquote>

<p>In the original Helmholtz Machine, the authors only proposed minimizing the Description Length for the wake phase, and was unable to achieve that (because they thought the gradient would have too much variance). Modern VAEs have reduced gradient variance significantly, giving us the freedom to directly optimize on the MDL objective.</p>

<p>In the Symmetric VAE, we propose minimizing both the wake-phase MDL and the sleep-phase MDL:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-923">\begin{align*}
&\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(x,z)}
\\=&\sum_x Q(x)KLD(Q(z|x)||P(z|x)) +\sum_xQ(x)log\frac{1}{P(x)}  + \\
     &\sum_z P(z)KLD(P(x|z)||Q(x|z))  +\sum_{z}P(z)log\frac{1}{Q(z)}
\end{align*}</script></p>

<p>This objective is an upper bound on the <script type="math/tex" id="MathJax-Element-924">-logP(x)</script> and <script type="math/tex" id="MathJax-Element-925">-logQ(z)</script>, which is exactly what we want. </p>

<p>Unfortunately, we cannot optimize the above objective directly, because it involves <script type="math/tex" id="MathJax-Element-926">Q(x,z)=Q(x)Q(z|x)</script>, which can be sampled from but cannot be computed. To optimize the above objective, we need to approximate <script type="math/tex" id="MathJax-Element-927">Q(x)</script>.</p>



<h3 id="approximating-qx">Approximating <script type="math/tex" id="MathJax-Element-928">Q(x)</script></h3>

<p>One method to approximately compute the MDL objective is to use an approximate <script type="math/tex" id="MathJax-Element-929">Q(x)</script>. We can train a normalized probability <script type="math/tex" id="MathJax-Element-930">Q'(x)</script> to approximate <script type="math/tex" id="MathJax-Element-931">Q(x)</script> by minimizing: <br>
<script type="math/tex; mode=display" id="MathJax-Element-932">\begin{align*}
      &\sum_x Q(x)log\frac{1}{Q'(x)}
\\= &KLD( Q(x) || Q'(x) ) + H(Q(x))
\end{align*}</script></p>

<p>We plug <script type="math/tex" id="MathJax-Element-933">Q'(x)</script> into the original objective to replace <script type="math/tex" id="MathJax-Element-934">Q(x)</script>, giving:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-935">\begin{align*}
     &\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(z|x)Q'(x)}
\\=&\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(z|x)Q(x)Q'(x)/Q(x)}
\\=&\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(x,z)} 
      +\sum_x P(x)log\frac{Q(x)}{Q'(x)}
\end{align*}</script></p>

<p>When <script type="math/tex" id="MathJax-Element-936">Q'(x)</script> and <script type="math/tex" id="MathJax-Element-937">Q(x)</script> are identical, we recover the original objective exactly.</p>



<h4 id="discriminator-loss">Discriminator Loss</h4>

<p>One complication with the above approach is that training with a normalized probability is difficult. So instead we train with <script type="math/tex" id="MathJax-Element-938">Q''(x)</script>, an unnormalized probability of <script type="math/tex" id="MathJax-Element-939">Q'(x)</script> <br>
<script type="math/tex; mode=display" id="MathJax-Element-940">
Q''(x) = Z Q'(x)
</script> <br>
where <script type="math/tex" id="MathJax-Element-941">Z=\sum_x Q''(x)</script>, which can estimated using <br>
<script type="math/tex; mode=display" id="MathJax-Element-942">\begin{align*}
Z =& \sum_x Q''(x)
\\=& \sum_x P(x|z) \frac{Q''(x)}{P(x|z)}
\\=& \sum_z P(z) \sum_x P(x|z) \frac{Q''(x)}{P(x|z)}
\\=& \sum_{x,z} P(x,z) \frac{Q''(x)}{P(x|z)}
\end{align*}</script></p>

<p>So we train <script type="math/tex" id="MathJax-Element-943">Q''(x)</script> to minimize:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-944">\begin{align*}
     & \sum_x Q(x) log\frac{1}{Q'(x)}
\\=& \sum_x Q(x) log\frac{1}{\frac{1}{Z}Q''(x)}
\\=& \sum_x Q(x) [log\frac{1}{Q''(x)} + logZ]
\end{align*}</script></p>

<p>Note that <script type="math/tex" id="MathJax-Element-945">(x,z)</script> in <script type="math/tex" id="MathJax-Element-946">Z</script> are sampled from <script type="math/tex" id="MathJax-Element-947">P</script>, and is completely independent from the <script type="math/tex" id="MathJax-Element-948">x</script> samples taken from <script type="math/tex" id="MathJax-Element-949">Q</script>. To reflect this independence and to avoid notational mix-up, we replace the <script type="math/tex" id="MathJax-Element-950">x</script> sampled from <script type="math/tex" id="MathJax-Element-951">Q(x)</script> with <script type="math/tex" id="MathJax-Element-952">v</script>. The above becomes: <br>
<script type="math/tex; mode=display" id="MathJax-Element-953">\begin{align*}
     &\sum_v Q(v) [log\frac{1}{Q''(v)} + log(\sum_{x,z}P(x,z)\frac{Q''(x)}{P(x|z)})]
\\=&E_{v \sim Q(v)}\left[ log\frac{1}{Q''(v)} + log(\sum_{x,z}P(x,z)\frac{Q''(x)}{P(x|z)}) \right]
\\=&E_{v \sim Q(v)}\left[ log\frac{1}{Q''(v)}  \right] - log(\sum_{x,z}P(x,z)\frac{P(x|z)}{Q''(x)}) 
\\\le & E_{v \sim Q(v)}\left[ log\frac{1}{Q''(v)}  \right] - E_{(x,z) \sim P(x,z)} \left[ log\frac{P(x|z)}{Q''(x)} \right]
\end{align*}</script></p>

<p>Because we only minimize the above w.r.t. parameters of <script type="math/tex" id="MathJax-Element-954">Q''(x)</script>, it is equivalent to optimizing:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-955">
E_{v \sim Q(v)}log\frac{1}{Q''(v)} - E_{x \sim P(x)}log\frac{1}{Q''(x)}
</script></p>

<p>Which is the discriminator loss in Generative Adversarial Networks. So <script type="math/tex" id="MathJax-Element-956">Q''(x)</script> is the discriminator.</p>



<h4 id="generator-gradient">Generator Gradient</h4>

<p>For the sake of demonstrating symmetric VAE’s connection with GANs, we can also obtain a gradient term similar to the generator gradient in GANs. Since GANs do not have the inference network <script type="math/tex" id="MathJax-Element-1460">Q(z|x)</script>, we remove all components of <script type="math/tex" id="MathJax-Element-1461">Q(z|x)</script>, and replace <script type="math/tex" id="MathJax-Element-1462">Q(x)</script> with <script type="math/tex" id="MathJax-Element-1463">Q'(x)</script>:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-1437">\begin{align*}
     &\sum_z P(z) \sum_x P(x|z)log\frac{P(x|z)}{Q'(x)}
\end{align*}</script></p>

<p>We differentiate <script type="math/tex" id="MathJax-Element-1438">log\frac{P(x|z)}{Q'(x)}</script> against the decoder parameter <script type="math/tex" id="MathJax-Element-1439">\phi</script> (the summations are Monte Carlo integration):</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-973">\begin{align*}
     &\nabla_\phi log\frac{P(x|z)}{Q'(x)}
\\=&  \nabla_\phi log\frac{1}{Q'(x)} 
       - \nabla_\phi log\frac{1}{P(x|z)}
\end{align*}</script></p>

<p>Note that in GANs, <script type="math/tex" id="MathJax-Element-974">P(x|z)=1</script>, but it isn’t the case for VAEs, which is why we have the second term above. The first term in the above can be written as: <br>
<script type="math/tex; mode=display" id="MathJax-Element-975">\begin{align*}
     & \nabla_x log \frac{1}{Q'(x)} \frac{\partial x}{\partial \phi}
\\=&[\nabla_x log \frac{1}{Q''(x)} + \nabla_x logZ]  \frac{\partial x}{\partial \phi}
\\=&\nabla_x log \frac{1}{Q''(x)} \frac{\partial x}{\partial \phi}
     && \text{$Z$ does not depend on $x$}
\end{align*}</script></p>

<p>Which is the generator gradient.</p>

<h1 id="do-we-really-need-a-discriminator">Do we really need a discriminator?</h1>

<p>Autoregressive models can generate sharp images without the use of a discriminator. This suggests that, under the right conditions, it is possible to get rid of the discriminator and still get high-quality models. But how do we do that with a latent variable model?</p>

<ul>
<li>Would multimodal likelihood and multimodal posterior help?</li>
<li>Would top-down attention (gradient descent as MCMC) help?</li>
</ul>

<p>I ask this question, because I can’t really find a process in human perception that’s equivalent to the discriminator, and we quite certainly do not tell the real from the fake while dreaming. Human perception largely relies on attention to sharpen things. But then again, humans aren’t that great at generating sharp images in the mind’s eye either. </p>

<p>On the other hand, when learning new motor actions, we do seem to use a discriminator (internal judgement and external critic). Also, while painting, we certainly use some internal judgement. Very interestingly, this “internal judgement” can be obtained solely through observation. For example, if I’ve seen a lot of van Gogh’s paintings, I’d be able to tell whether a painting looks like it’s done by him, even if I’ve never used a paintbrush in my life.</p>

<p>This “internal judgement”, which can be obtained through observation alone without generation, leads me to thinking: can the recognition model itself serve as the discriminator? That is, can we replace <script type="math/tex" id="MathJax-Element-1430">Q'(x)</script> with <script type="math/tex" id="MathJax-Element-1431">P(x)</script>, and <script type="math/tex" id="MathJax-Element-1432">\nabla_xlog\frac{1}{Q'(x)}</script> with <script type="math/tex" id="MathJax-Element-1433">\nabla_x\sum_z Q(z|x)log\frac{Q(z|x)}{P(x,z)}</script>? Also, what if P and Q have tied weights? This is starting to look like attention to me.</p>

Please comment!
