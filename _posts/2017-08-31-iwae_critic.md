---
layout: post
title:  "Importance-weighted update with critic"
date:   2017-08-31 16:00:00 +0800
permalink: /blog/iwae_critic
comments: true
categories: jekyll update
---


<p>In this post we provide a variance reduction trick for variational autoencoders that uses importance-weighted updates in a way similar to Importance-weighted autoencoder [1]. However, the technique we propose here uses a critic, and only uses a single sample. So it might be a bit more efficient than IWAE.</p>

<p>We begin with the derivation of importance-weighted autoencoder. </p>

<p>The K-sample estimator of <script type="math/tex" id="MathJax-Element-904">logP(x)</script>:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-405">\begin{align*}
     & logP(x)
\\=& log \sum_z Q(z|x) \frac{P(x,z)}{Q(z|x)}
      &&\text{importance sampling with approximate posterior}
\\=& log E_{z^{(i)} \sim Q(z|x)}  \sum_i^K  \frac{1}{K}  \frac{P(x,z^{(i)})}{Q(z^{(i)}|x)}
      &&\text{K-sample estimator of above}
\\\ge & E_{z^{(i)} \sim Q(z|x)} log \sum_i^K  \frac{1}{K}  \frac{P(x,z^{(i)})}{Q(z^{(i)}|x)}
       && \text{Jensen's inequality}
\end{align*}</script></p>

<p>To simplify notation, we denote <script type="math/tex" id="MathJax-Element-406">\frac{P(x,z^{(i)})}{Q(z^{(i)}|x)}</script> as <script type="math/tex" id="MathJax-Element-407">w^{(i)}</script>, the above is:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-259">
E_{z^{(i)} \sim Q(z|x)} log \sum_i^K  \frac{1}{K}  w^{(i)}
</script></p>

<p>Its gradient w.r.t. <script type="math/tex" id="MathJax-Element-260">\theta</script>:</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-1119">\begin{align*}
     &E_{z^{(i)} \sim Q(z|x)}  \nabla_\theta log \sum_i^K  \frac{1}{K}  w^{(i)}
\\=&E_{z^{(i)} \sim Q(z|x)}   \frac{\sum_i^K \nabla_\theta  w^{(i)} } {\sum_i^K w^{(i)}}
\\=&E_{z^{(i)} \sim Q(z|x)}   \frac{\sum_i^K w^{(i)}  \nabla_\theta  log w^{(i)} } {\sum_i^K w^{(i)}}
\end{align*}</script></p>

<blockquote>
  <p>At this point, it helps to note that <script type="math/tex" id="MathJax-Element-1120">w^{(i)}</script> isn’t a term we would normally compute. Usually we would compute <script type="math/tex" id="MathJax-Element-1121">log\frac{P(x,z)}{Q(z|x)}</script> in a VAE, and <script type="math/tex" id="MathJax-Element-1122">w^{(i)}</script> is the exponential of that value. However, you shouldn’t actually compute that exponential because it’ll just be 0 (the exponential of a very negative value is basically 0). However, the big fraction still works out because it actually forms a softmax, which is computed by first subtracting the minimal value then taking expoential. So it is obvious that the gradient computed from the  <script type="math/tex" id="MathJax-Element-1123">i</script>th sample <script type="math/tex" id="MathJax-Element-1124">\nabla_\theta logw^{(i)}</script> is weighted by its corresponding softmax value.</p>
</blockquote>

<p>The above is pretty much what importance-weighted autoencoder is all about. Now, time for our little trick.</p>

<p>We continue with the K-sample gradient estimator. When K is very large, it becomes:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-1273">\begin{align*}
    & E_{z^{(i)} \sim Q(z|x)}   \frac{\sum_i^K w^{(i)}  \nabla_\theta  log w^{(i)} } {K E_{z\sim Q(z|x)}w(z)}
\end{align*}</script></p>

<p>If we set <script type="math/tex" id="MathJax-Element-1274">K=1</script>, no bias is introduced. The above becomes:</p>

<p><script type="math/tex; mode=display" id="MathJax-Element-1244">\begin{align*}
     & E_{z^{(1)} \sim Q(z|x)}   \frac{w^{(1)}  } {E_{z\sim Q(z|x)}w(z)} \nabla_\theta  log w^{(1)} 
\\=& E_{z^{(1)} \sim Q(z|x)}   e^{   log w^{(1)} - logE_{z\sim Q(z|x)} w(z) } \nabla_\theta  log w^{(1)} 
\\\le & E_{z^{(1)} \sim Q(z|x)}   e^{   log w^{(1)} - E_{z\sim Q(z|x)} log w(z) } \nabla_\theta  log w^{(1)} 
\\\approx & E_{z^{(1)} \sim Q(z|x)}   e^{   log w^{(1)} - C(x) } \nabla_\theta  log w^{(1)} 
\end{align*}</script></p>

<p>Where <script type="math/tex" id="MathJax-Element-1245">C(x)</script>, the critic, is trained to approximate <script type="math/tex" id="MathJax-Element-1246">E_{z\sim Q(z|x)} log w(z)</script>.</p>

<p>[1] Yura Burda et al., Importance weighted autoencoders</p>
