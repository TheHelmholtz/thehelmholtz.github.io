---
layout: page
title: About
footer: true
permalink: /about/
---

I am Hou Yunqing. Hou is my family name.
{% include icon-github.html username="hyqneuron" %} {% include icon-linkedin.html pagename="hou-yunqing" %}. 

I care about 3 things:

- Neural nets and unsupervised learning
- Assembly-level optimization of GPU code
- Modular programming languages


#### Neural Nets

I like unsupervised methods that preserve information. VAEs and Boltzmann machines are examples of such methods. These
methods currently don't scale to bigger datasets, and I've been trying to figure out why. Some of my forward-looking
research interests are listed [here](/direction).

#### GPU assembly coding

I wrote an [assembler](http://code.google.com/p/asfermi) for NVIDIA Fermi GPUs back in 2011, while I was trying to write
more optimized GPU code for neural nets. I'm now starting another project, with the objective of enabling real time
object detection, semantic segmentation and depth estimation within a small power package, using optimization at both
the algorithmic and the assembly level.

#### Modular programming languages

The basic idea is that we have a base language, on which new language features can be added as modules. The difficult
thing is that it's impossible to have unconstrained extensibility without breaking semantic consistency (add too many
extensions and your language will go nuts). I've been exploring a method that puts some constraints on extensibility so
as to trade for the ability to statically verify correctness of the extended language.

