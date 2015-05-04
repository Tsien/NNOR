### MSEB+EML+PSA

Author: Tsien
Modified Date: 05/04/2015
The target of this experiment is to combine ELM with MSEB.

#### Main Idea:
1. use new convex objective function MSEB[1]
2. use Extreme Learning Machine framework
3. use Principle Sensitivity Analysis to prune

---

#### About Extreme learning machine(ELM):
There are two kinds of weights. 

1. input weights of an SLFN can be randomly chosen(according to any continuous sampling distribution)
2. output weights of an SLFN can be analytically determined by the minimum norm least-squares solutions.

Based on NewConvexFunc, add an additional output layer. The parameters of this layer are analytically determined.

