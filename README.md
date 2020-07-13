# Multiple Aspect Learning GSOM (MAL-GSOM)

The GSOM consists of
* AspectLearner GSOM - The base layer GSOM that learn aspects individually
* Associative GSOM - The hierarchy layer GSOM that combine aspects and learn

Parameters
---
* *aspect* (miniumum =2) = number of aspects to be learneed (No. of base layer GSOMs)
* *hierarchy* (minimum =1, Default *ceil(aspect/2)*)= number of hierarchies created for associative GSOM to learn aspects in combination.

Samle algorithm flow (aspect=2,hierarchy=1)
---
* aspect =2
* hierarchy (default) =1

![sample_algorithm](https://github.com/VivekVinushanth/Parallel-GSOM/blob/master/Diagrams/MAL-GSOM.png)
