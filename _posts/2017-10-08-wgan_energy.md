---
layout: post
title:  "WGANs are energy-based"
date:   2017-10-06 12:00:00 +0800
permalink: /blog/wgan_energy
comments: true
categories: jekyll update
---



The discriminator used in WGANs, trained without sigmoid, is actually an energy-based approximator for the marginal
probability of data. That is, $$D(x)$$ is approximately proportional to $$logP_{real}(x)$$. To people familiar with
energy-based models this is a pretty obvious idea, but not everybody finds it intuitive, so here's the derivation.

I really hate typing out $$P_{real}(x)$$, So I'll write it as $$Q(x)$$ instead. For the same reason, I'll abbreviate the
generator's generated distribution, $$P_{fake}(x)$$, as just $$P(x)$$.

# Approximate $$Q(x)$$
Let's start by trying to approximate the real data distribution, $$Q(x)$$. Suppose the approximator we use is a neural
network, that takes $$x$$ as input, and outputs a single unnormalized probability value. Let's call this approximator
$$Q'(x)$$. Now, $$Q'(x)$$ is unnormalized. To make things tick, we'd want to convert it to a normalized probability first.
To do that, we'd have to compute its normalizing constant:

$$
Z = \sum_x Q'(x)
$$

Of course this summation is intractable, as there are too many values of $$x$$ to sum over. However, we can estimate its
value using importance sampling:

$$
\begin{align*}
Z &= \sum_x Q'(x) \\
   &= \sum_x P(x) \frac{Q'(x)}{P(x)} \\
   &= \mathbf{E}_{x\sim P} \frac{Q'(x)}{P(x)}
\end{align*}
$$

The more closely $$P(x)$$ matches the normalized version of $$Q'(x)$$, the more efficient the above estimator will be. If
you replace $$P(x)$$ with some random distribution, the above estimator might take forever to converge. 

So now we have the normalizing constant, the normalized distribution is just:
$$
\frac{1}{Z} Q'(x)
$$

We can use the above normalized distribution to approximate $$Q(x)$$, by minimizing the cross-entropy:

$$
\begin{align*}
  &\sum_x Q(x) log\frac{1}{ \frac{1}{Z}Q'(x) } \\
=& \mathbf{E}_{x\sim Q}log\frac{1}{Q'(x)} + logZ \\
=& \mathbf{E}_{x\sim Q}log\frac{1}{Q'(x)} + log\mathbf{E}_{x\sim P} \frac{Q'(x)}{P(x)} \\
=& \mathbf{E}_{x\sim Q}log\frac{1}{Q'(x)} - log\mathbf{E}_{x\sim P} \frac{P(x)}{Q'(x)} 
        && \text{invert $P$ and $Q'$ in fraction}\\
\le & \mathbf{E}_{x\sim Q}log\frac{1}{Q'(x)} - \mathbf{E}_{x\sim P} log\frac{P(x)}{Q'(x)} 
        && \text{Jensen's inequality}
\end{align*}
$$

We minimize this cross-entropy w.r.t. the parameters of $$Q'(x)$$. Since $$P(x)$$ is irrelevant as long as gradient w.r.t.
$$Q'(x)$$'s parameters is concerned, we can drop $$P(x)$$ from the objective without changing the gradient, giving:

$$
\mathbf{E}_{x\sim Q}log\frac{1}{Q'(x)} - \mathbf{E}_{x\sim P} log\frac{1}{Q'(x)} 
$$

If we write $$D(x) = logQ'(x)$$, the above is:

$$
-\mathbf{E}_{x\sim Q}D(x) + \mathbf{E}_{x\sim P} D(x)
$$

Which means we just maximize the discriminator's output for $$x_{real}$$, and minimize the discriminator output for
$$x_{fake}$$. This is exactly the same as the WGAN objective for discriminator.

# Some notes

Strictly speaking, we don't need to sample from $$P(x)$$ in order to estimate $$Z$$. But we need a distribution that closely
matches the normalized $$Q'(x)$$. The generator's distribution is probably the closest we've got. One possible alternative
might be to sample directly from normalized $$Q'(x)$$, but that''ll be a lot more expensive as it'll certainly involve
MCMC.
