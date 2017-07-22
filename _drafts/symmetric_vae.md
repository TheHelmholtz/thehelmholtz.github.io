---
layout: post
title:  "Symmetric VAE derives GAN"
date:   2017-07-15 23:00:00 +0800
permalink: /blog/symmetric_vae
comments: true
categories: jekyll update
---


In this post, we start by discussing the asymmetry of traditional Variational Autoencoders, followed by a short
review of Boltzmann Machine and Helmholtz Machine, both of which use a symmetric cost. We then introduce the the
Extended Helmholtz Machine, a type of VAE with a symmetric cost, and demonstrate how GAN can be derived from
it.

<blockquote>
  <p>A note on notation: in this post, most of the summation actually imply expectation, so gradient is not taken on
  them. We write expectation as summation because the LaTeX code of expectation is quite unreadable. Also, all
  lower-case letters are vectors</p>
</blockquote>


<h1 id="vaes-asymmetry">VAE’s asymmetry</h1>

<p>The objective of VAE is:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-927">\begin{align*}
     &\sum_x Q(x) \sum_z Q(z|x)log\frac{Q(z|x)}{P(x,z)}
\\=&\sum_x Q(x)[KLD( Q(z|x) || P(z|x) ) + log\frac{1}{P(x)}]
\end{align*}</script></p>

<p>Note that <script type="math/tex" id="MathJax-Element-928">Q(x)</script> is the data distribution, <script type="math/tex" id="MathJax-Element-929">Q(z|x)</script> the approximate posterior, and <script type="math/tex" id="MathJax-Element-930">P</script> the generative model.</p>

<p>VAE is asymmetric in the sense that sampling is performed in only one direction (from <script type="math/tex" id="MathJax-Element-931">Q</script>). In comparison, both Boltzmann Machine and Helmholtz Machine perform sampling from both <script type="math/tex" id="MathJax-Element-932">Q</script> (real data) and <script type="math/tex" id="MathJax-Element-933">P</script> (fantasy data). The disadvantage of sampling from only <script type="math/tex" id="MathJax-Element-934">Q</script> is that there will be regions in <script type="math/tex" id="MathJax-Element-935">(x,z)</script> whose probability under <script type="math/tex" id="MathJax-Element-936">Q</script> is zero, yet whose probability under <script type="math/tex" id="MathJax-Element-937">P</script> is greater than zero.</p>

<p><img src="https://lh3.googleusercontent.com/-G6nSM2ag-oU/WVe04A09biI/AAAAAAAADgw/SP8_NeS-vRE6Ckyf-hIGn-B4YBxHfVDggCLcBGAs/s0/Q_and_P.png" alt="enter image description here" title="Q_and_P.png"></p>

<p>The image above illustrate this possibility. Because training is performed by sampling exclusively from <script type="math/tex" id="MathJax-Element-938">Q</script>, the regions that are not covered by <script type="math/tex" id="MathJax-Element-939">Q</script> may misbehave under <script type="math/tex" id="MathJax-Element-940">P</script>.</p>

<blockquote>
  <p>An intuitive interpretation: consider the case where a teacher is teaching a student to solve some problems. The teacher has limited time, so he only explains a subset of all possible problems. Let’s call the problems that the teacher has shown to the student the “taught problems”, and the rest “untaught problems”. </p>
  
  <p>If the student blindly accepts whatever the teacher teaches, he will not be able to handle untaught problems that are different from the taught ones. However, if the student is very curious, and frequently asks questions, and the teacher in turn gives answers, then we can reasonably expect this curious student to be much better at solving the untaught problems.</p>
  
  <p>When sampling from <script type="math/tex" id="MathJax-Element-941">Q</script>, the teacher is teaching the “taught problems”. When sampling from <script type="math/tex" id="MathJax-Element-942">P</script>, the student is asking questions about “untaught problems”.</p>
</blockquote>

<p>In the next part, we briefly review the Boltzmann Machine and the Helmholtz Machine, both of which optimize a symmetric objective.</p>



<h1 id="symmetric-objectives">Symmetric objectives</h1>



<h2 id="boltzmann-machine">Boltzmann Machine</h2>

<p>Boltzmann Machine maximizes the following objective:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-943">
          \frac{\sum_h U(v,h)}{\sum_{x,z}U(x,z)}
       = \frac{\sum_h \frac{1}{Z}U(v,h)}{\sum_{x,z}\frac{1}{Z}U(x,z)}
       = \frac{\sum_h P(v,h)}{\sum_{x,z}P(x,z)}
       = \frac{P(v)}{1}
</script></p>

<ul>
<li>Positive phase: maximize probability of observed data</li>
<li>Negative phase: minimize probability of fantasy data</li>
<li>Notation:  <br>
<ul><li><script type="math/tex" id="MathJax-Element-944">U</script> is unnormalized probability</li>
<li><script type="math/tex" id="MathJax-Element-945">v</script> is observed data, <script type="math/tex" id="MathJax-Element-946">h</script> is the corresponding latent state sample</li>
<li><script type="math/tex" id="MathJax-Element-947">x</script> and <script type="math/tex" id="MathJax-Element-948">z</script> are visible and latent samples of the fantasy distribution</li></ul></li>
</ul>

<p>The symmetry is obvious in the positive phase and the negative phase.</p>



<h2 id="helmholtz-machine">Helmholtz Machine</h2>

<p>The Wake-Sleep Algorithm was used to train the first Helmholtz Machine, which has a recognition model <script type="math/tex" id="MathJax-Element-949">Q</script> and a generative model <script type="math/tex" id="MathJax-Element-950">P</script>. It is very similar to modern VAEs, but optimizes a different objective:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-951">\begin{align*}
&\sum_{x,z}Q(x,z)log\frac{1}{P(x|z)} + \sum_{x,z}P(x,z)log\frac{1}{Q(z|x)}
\\=&\sum_{x,z}Q(x,z)log\frac{Q(x|z)}{P(x|z)Q(x|z)} + \sum_{x,z}P(x,z)log\frac{P(z|x)}{Q(z|x)P(z|x)}
\\=&\sum_z Q(z)[KLD( Q(x|z) || P(x|z) ) + H(Q(x|z))] + \\
     &\sum_x P(x)[KLD( P(z|x) || Q(z|x) ) + H(P(z|x))]
\end{align*}</script></p>

<ul>
<li>Wake phase: bring <script type="math/tex" id="MathJax-Element-952">P</script> to <script type="math/tex" id="MathJax-Element-953">Q</script></li>
<li>Sleep phase: bring <script type="math/tex" id="MathJax-Element-954">Q</script> to <script type="math/tex" id="MathJax-Element-955">P</script></li>
</ul>

<p>Again, the symmetry is obvious in the wake phase and the sleep phase.</p>

<p>There are two problems associated with traditional Helmholtz Machines</p>

<ol>
<li>It does not optimize the proper Minimum Description Length objective</li>
<li>It cannot handle multimodal posterior</li>
</ol>

<p>In the Symmetric VAE, our aim is to tackle both problems. In fact, the Symmetric VAE can be considered as a form of extended Helmholtz Machine.</p>

<p>We begin by giving the Helmholtz Machine a proper Minimum Description Length objective.</p>



<h1 id="symmetric-vae">Symmetric VAE</h1>

<p>Consider the information flow in both the wake phase and the sleep phase:</p>

<p>The wake phase: <br>
<script type="math/tex; mode=display" id="MathJax-Element-956"> x \longrightarrow z \longrightarrow z' \longrightarrow x'</script> <br>
The sleep phase: <br>
<script type="math/tex; mode=display" id="MathJax-Element-957"> z' \longrightarrow x' \longrightarrow x \longrightarrow z</script></p>

<p>In the original Helmholtz Machine, the authors only proposed minimizing the Description Length for the wake phase, and was unable to achieve that (because they thought the gradient would have too much variance). Modern VAEs have reduced gradient variance significantly, giving us the freedom to directly optimize on the MDL objective.</p>

<p>In the Symmetric VAE, we propose minimizing both the wake-phase MDL and the sleep-phase MDL:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-958">\begin{align*}
&\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(x,z)}
\\=&\sum_x Q(x)KLD(Q(z|x)||P(z|x)) +\sum_xQ(x)log\frac{1}{P(x)}  + \\
     &\sum_z P(z)KLD(P(x|z)||Q(x|z))  +\sum_{z}P(z)log\frac{1}{Q(z)}
\end{align*}</script></p>

<p>This objective is an upper bound on the <script type="math/tex" id="MathJax-Element-959">-logP(x)</script> and <script type="math/tex" id="MathJax-Element-960">-logQ(z)</script>, which is exactly what we want. </p>

<p>Unfortunately, we cannot optimize the above objective directly, because it involves <script type="math/tex" id="MathJax-Element-961">Q(x,z)=Q(x)Q(z|x)</script>, which can be sampled from but cannot be computed. To circumvent this, we optimize a slight different objective.</p>

<p>We begin by replacing <script type="math/tex" id="MathJax-Element-962">Q(x,z)</script> with <script type="math/tex" id="MathJax-Element-963">Q(z|x)</script>:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-964">\begin{align*}
&\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(z|x)}
\\=&\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(x,z)/Q(x)}
\\=&\sum_x Q(x)KLD(Q(z|x)||P(z|x)) + \sum_xQ(x)log\frac{1}{P(x)} +  \\
     &\sum_z P(z)KLD(P(x|z)||Q(x|z))  +\sum_{z}P(z)log\frac{1}{Q(z)}
      -\sum_{x}P(x)log\frac{1}{Q(x)}
\end{align*}</script></p>

<p>Because the additional <script type="math/tex" id="MathJax-Element-965">log\frac{1}{Q(x)}</script> has no parameters (data distribution), it has no gradient. Consequently, this alternative objective produces the same gradient as the original objective.</p>

<p>In the case that we must obtain an approximate value of the original objective, we can adopt one of two methods:</p>

<ol>
<li>Multiply the wake-phase loss by 2, or</li>
<li>Use an approximator of <script type="math/tex" id="MathJax-Element-966">Q(x)</script>, which gives rise to a discriminator</li>
</ol>



<h3 id="unbalanced-estimation">Unbalanced Estimation</h3>

<p>In this method, we multiply the wake-phase loss by 2, thus computing:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-967">\begin{align*}
 2&\sum_x Q(x)KLD(Q(z|x)||P(z|x)) +\sum_xQ(x)log\frac{1}{P(x)} + \\
   &\sum_z P(z)KLD(P(x|z)||Q(x|z)) +\sum_{z}P(z)log\frac{1}{Q(z)} + \\
   &[\sum_xQ(x)log\frac{1}{P(x)} -\sum_{x}P(x)log\frac{1}{Q(x)}]
\end{align*}</script></p>

<p>At convergence, <script type="math/tex" id="MathJax-Element-968">P</script> and <script type="math/tex" id="MathJax-Element-969">Q</script> would be close, so we expect the two cross-entropy terms inside the bracket to have a small value, thus giving us an approximation on the total descriptive length.</p>



<h3 id="approximating-qx">Approximating <script type="math/tex" id="MathJax-Element-970">Q(x)</script></h3>

<p>Another method to approximately compute the original objective is to use an approximate <script type="math/tex" id="MathJax-Element-971">Q(x)</script>. We can train a normalized probability <script type="math/tex" id="MathJax-Element-972">Q'(x)</script> to approximate <script type="math/tex" id="MathJax-Element-973">Q(x)</script> by minimizing: <br>
<script type="math/tex; mode=display" id="MathJax-Element-974">\begin{align*}
      &\sum_x Q(x)log\frac{1}{Q'(x)}
\\= &KLD( Q(x) || Q'(x) ) + H(Q(x))
\end{align*}</script></p>

<p>We plug <script type="math/tex" id="MathJax-Element-975">Q'(x)</script> into the original objective to replace <script type="math/tex" id="MathJax-Element-976">Q(x)</script>, giving:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-977">\begin{align*}
     &\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(z|x)Q'(x)}
\\=&\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(z|x)Q(x)Q'(x)/Q(x)}
\\=&\sum_{x,z}Q(x,z)log\frac{Q(z|x)}{P(x,z)} + \sum_{x,z}P(x,z)log\frac{P(x|z)}{Q(x,z)} 
      +\sum_x P(x)log\frac{Q(x)}{Q'(x)}
\end{align*}</script></p>

<p>When <script type="math/tex" id="MathJax-Element-978">Q'(x)</script> and <script type="math/tex" id="MathJax-Element-979">Q(x)</script> are identical, we recover the original objective exactly.</p>



<h4 id="discriminator-loss">Discriminator Loss</h4>

<p>One complication with the above approach is that training with a normalized probability is difficult. So instead we train with <script type="math/tex" id="MathJax-Element-980">Q''(x)</script>, an unnormalized probability of <script type="math/tex" id="MathJax-Element-981">Q'(x)</script> <br>
<script type="math/tex; mode=display" id="MathJax-Element-982">
Q''(x) = Z Q'(x)
</script> <br>
where <script type="math/tex" id="MathJax-Element-983">Z=\sum_x Q''(x)</script>, which can estimated using <br>
<script type="math/tex; mode=display" id="MathJax-Element-984">\begin{align*}
Z =& \sum_x Q''(x)
\\=& \sum_x P(x|z) \frac{Q''(x)}{P(x|z)}
\\=& \sum_z P(z) \sum_x P(x|z) \frac{Q''(x)}{P(x|z)}
\\=& \sum_{x,z} P(x,z) \frac{Q''(x)}{P(x|z)}
\end{align*}</script></p>

<p>So we train <script type="math/tex" id="MathJax-Element-985">Q''(x)</script> to minimize:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-986">\begin{align*}
     & \sum_x Q(x) log\frac{1}{Q'(x)}
\\=& \sum_x Q(x) log\frac{1}{\frac{1}{Z}Q''(x)}
\\=& \sum_x Q(x) [log\frac{1}{Q''(x)} + logZ]
\end{align*}</script></p>

<p>Note that <script type="math/tex" id="MathJax-Element-987">(x,z)</script> in <script type="math/tex" id="MathJax-Element-988">Z</script> are sampled from <script type="math/tex" id="MathJax-Element-989">P</script>, and is completely independent from the <script type="math/tex" id="MathJax-Element-990">x</script> samples taken from <script type="math/tex" id="MathJax-Element-991">Q</script>. To reflect this independence and to avoid notational mix-up, we replace the <script type="math/tex" id="MathJax-Element-992">x</script> sampled from <script type="math/tex" id="MathJax-Element-993">Q(x)</script> with <script type="math/tex" id="MathJax-Element-994">v</script>. The above becomes: <br>
<script type="math/tex; mode=display" id="MathJax-Element-995">\begin{align*}
     &\sum_v Q(v) [log\frac{1}{Q''(v)} + log(\sum_{x,z}P(x,z)\frac{Q''(x)}{P(x|z)})]
\\=&E_{v \sim Q(v)}\left[ log\frac{1}{Q''(v)} + log(\sum_{x,z}P(x,z)\frac{Q''(x)}{P(x|z)}) \right]
\\=&E_{v \sim Q(v)}\left[ log\frac{1}{Q''(v)}  \right] - log(\sum_{x,z}P(x,z)\frac{P(x|z)}{Q''(x)}) 
\\\le & E_{v \sim Q(v)}\left[ log\frac{1}{Q''(v)}  \right] - E_{(x,z) \sim P(x,z)} \left[ log\frac{P(x|z)}{Q''(x)} \right]
\end{align*}</script></p>

<p>Because we only minimize the above w.r.t. parameters of <script type="math/tex" id="MathJax-Element-996">Q''(x)</script>, it is equivalent to optimizing:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-997">
E_{v \sim Q(v)}log\frac{1}{Q''(v)} - E_{x \sim P(x)}log\frac{1}{Q''(x)}
</script></p>

<p>Which is the discriminator loss in Generative Adversarial Networks. So <script type="math/tex" id="MathJax-Element-998">Q''(x)</script> is the discriminator.</p>



<h4 id="generator-gradient">Generator Gradient</h4>

<p>Similarly, we can derive the generator gradient. Since GANs do not have the inference network <script type="math/tex" id="MathJax-Element-999">Q(z|x)</script>, we remove all components of <script type="math/tex" id="MathJax-Element-1000">Q(z|x)</script>, and replace <script type="math/tex" id="MathJax-Element-1001">Q(x)</script> with <script type="math/tex" id="MathJax-Element-1002">Q'(x)</script>:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-1003">\begin{align*}
     &\sum_z P(z) \sum_x P(x|z)log\frac{P(x|z)}{Q'(x)}
\end{align*}</script></p>

<p>We differentiate <script type="math/tex" id="MathJax-Element-1004">log\frac{P(x|z)}{Q'(x)}</script> against the decoder parameter <script type="math/tex" id="MathJax-Element-1005">\phi</script> (the other terms form an expectation):</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-1006">\begin{align*}
     &\nabla_\phi log\frac{P(x|z)}{Q'(x)}
\\=&  \nabla_\phi log\frac{1}{Q'(x)} 
       - \nabla_\phi log\frac{1}{P(x|z)}
\end{align*}</script></p>

<p>Note that in GANs, <script type="math/tex" id="MathJax-Element-1007">P(x|z)=1</script>, but it isn’t the case for VAEs, which is why we have the second term above. The first term in the above can be written as: <br>
<script type="math/tex; mode=display" id="MathJax-Element-1008">\begin{align*}
     & \nabla_x log \frac{1}{Q'(x)} \frac{\partial x}{\partial \phi}
\\=&[\nabla_x log \frac{1}{Q''(x)} + \nabla_x logZ]  \frac{\partial x}{\partial \phi}
\\=&\nabla_x log \frac{1}{Q''(x)} \frac{\partial x}{\partial \phi}
     && \text{$Z$ does not depend on $x$}
\end{align*}</script></p>

<p>Which is the generator gradient. The conclusion is that Generative Adversarial Network is an extremely stripped-down version of the Symmetric VAE, with an additional discriminator that approximates <script type="math/tex" id="MathJax-Element-1009">Q(x)</script>.</p>
