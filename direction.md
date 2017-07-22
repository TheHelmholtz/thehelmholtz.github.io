---
layout: page
title: Research Direction
permalink: /direction/
---

I am very passionate about these topics:

* Approximate + Exact Inference
* Discrete Entity Modelling
* Attention and Memory
* Binding


#### Approximate + Exact Inference

Neural nets are now quite good at producing *intuition*. I believe the next step is to combine this powerful intuition
with the accuracy of exact inference methods.


#### Discrete Entity Modelling

The world we observe usually consists of a large set of discrete objects, yet the representation we use in our NNs are
primarily unimodal. To model the discrete structure of our world, we must move towards representations that are
multimodal, where each mode may represent a separate object.


#### Attention and Memory

Inference with multimodal distributions is usually hard. I believe attention, implemented as a kind of noisy gradient
descent (MCMC with Langevin dynamics), has the ability to handle multimodal posteriors. In addition, through the act of
binding, attention allows very complex inference to be performed with a very limited set of computational resources
(sequential execution).

Memory, on the other hand, is something we cannot do without if we want learning to be efficient. Our current models
work reasonablly well at capturing population-level statistics. However, retaining instance-level information is an
ability that is still lacking. It is widely understood that memory can be modelled using a multimodal distribution,
every mode of which is a memory segment. For this reason, memory is closely tied to attention, because memory recall is
essentially an inference process on a multimodal distribution.


#### Binding

Binding is the unconscious act of associating signals from multiple sensory modality and across time to the same
underlying "object identity". The binding problem is interesting to me because it involves the modelling of multiple
entities in the same scene, and is an excellent testing ground for discrete entity modelling methods.

