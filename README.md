# Multiple Aspect Learning GSOM (MAL-GSOM)

The GSOM consists of
* AspectLearner GSOM - The base layer GSOM that learn aspects individually
* Associative GSOM - The hierarchy layer GSOM that combine aspects and learn

Parameters
---
* *aspect* (miniumum =2) = number of aspects to be learneed (No. of base layer GSOMs)
* *hierarchy* (minimum =1, Default *ceil(aspect/2)*)= number of hierarchies created for associative GSOM to learn aspects in combination.

Samle
---
* aspect =2
* hierarchy (default) =2

![sample_algorithm](https://drive.google.com/file/d/1MqI_E9yjtxen34LN-AEzByMwfhjYzpIp/view?usp=sharing)
