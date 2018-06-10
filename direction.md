---
layout: page
title: Research Interests
footer: true
permalink: /direction/
---

I am no longer actively doing research, but I still enjoy thinking about how to model the world with unsupervised
methods.

A few things that I think haven't entered mainstream unsupervised methods:

- Model: model that accomodates an unlimited set of object entities (or event episodes)
- Inference: iterative and active inference
- Learning: symmetric learning (e.g. [this](/blog/symmetric_vae))


## Forward-looking topics

#### Discrete Entity Modelling

The world we observe usually consists of a large set of discrete objects, yet the representation we use in our NNs are
primarily unimodal. To model the discrete structure of our world, we must move towards representations that are
multimodal, where each mode may represent a separate object.


#### Attention and Memory

Inference with multimodal distributions can be tricky. I believe attention, implemented as a kind of noisy gradient
descent (MCMC with Langevin dynamics), has the potential to handle multimodal posteriors.

Memory, on the other hand, is something we need if we want learning to be efficient. Our current models work reasonablly
well at capturing population-level statistics. However, retaining instance-level information is an ability that is still
lacking. It is widely understood that memory can be modelled using a multimodal distribution, every mode of which is a
memory segment. For this reason, memory is closely tied to attention, because memory recall is essentially an inference
process on a multimodal distribution.


#### Binding

Binding is the unconscious act of associating signals from multiple sensory modality and across time to the same
underlying "object identity". The binding problem is interesting to me because it involves the modelling of multiple
entities in the same scene, and is an excellent testing ground for discrete entity modelling methods. Also, by combining
signals from multiple sensory modalities and across time, we can make the recognition of "object identity" much more
robust to variations in specific modalities.

