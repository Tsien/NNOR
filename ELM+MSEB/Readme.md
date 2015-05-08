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

#### Algorithm

1. 数据预处理:
    * 把标签改为CA编码
    * 把01CA编码改为0.95，0.05
2. 根据经验公式确定隐藏节点个数的候选
3. 按照隐藏节点个数建立网络
4. 10折交叉验证，挑选出在验证集上效果最好的那个模型
5. 重复第4步共P遍，获得P个基学习机
6. 集成P个基学习机
