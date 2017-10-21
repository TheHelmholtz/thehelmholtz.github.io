---
layout: page
title: Research Interests [New]
footer: false
permalink: /direction2/
---

## Current topics

Mostly I study unsupervised learning in the context of artificial neural networks. I'm trying to figure out why
Variational Autoencoders can't scale to bigger datasets. Currently some of the immediate things that interest me
include:

- Hierarchical latent code
- Iterative inference
- Temporal structures 
- Scene-level structures

#### Hierarchical latent code

VAEs without a hierarchical latent code cannot possibly scale to complex datasets. A single top-level latent code can
only retain very limited amount of information, making a reconstruction loss rather hopeless. It makes sense to go with
a hierarchical latent code, but it turns out that getting a hierarchical latent code to work is not nearly as easy.

Currently, most of the hierarchical conv-VAE filters I've trained (on CIFAR10) are qualitatively very
different from supervised CNNs trained on the same dataset. In particular, supervised CNNs learn filters that are
sensitive to *sharp object outlines*, whereas filters in hierarchical conv-VAEs tend to be more sensitive to colour
instead of outline.

#### Iterative inference

TBD

#### Temporal structures

TBD

#### Scene-level structures

TBD

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

