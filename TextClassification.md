《A Comparison of Event Models for Naive Bayes Text Classification》

设mixture model的components是c，参数为theta，documents是d，words是w，词汇表是V，则一个document的似然值为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/181.png)
 
假设每个mixture model component对应一个class；

(1)	Multi-variate Bernoulli Model

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/182.png)

B为0或1，表明单词w(t)是否出现在文本d(i)中，

使用训练样本来估计w在class c中的出现的概率：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/183.png)

P(c|d)为0或1，由document d的class决定，

class的先验参数，通过极大似然估计得到：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/184.png)

Multi-variate Bernoulli Model只考虑word是否出现，不考虑word出现的次数，没有出现的word也要把不出现的概率计算进去；

(2)	Multinomial Model

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/185.png)

N(it)是word t出现在document d中的次数，

w在class c中的出现的概率为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/186.png)

Class的先验概率和Multi-variate Bernoulli Model相同；

(3)	Classification

使用Bayes公式：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/187.png)

所以这种分类模型是generative model。

(4)	Feature Selection

选择average mutual information最高的words来作为特征。设C是所有classes的随机变量，W(t)取值0或1，表明word w(t)是否出现，H(C)是class变量的熵，H(C|W(t))是在word出现或不出现的条件下class变量的熵，则average mutual information值为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/188.png)

multi-variate Bernoulli model的计算：P(c)是所有class c的documents的个数除以documents的总数，P(f(t))是word w(t)出现的documents的个数除以documents的总数，P(c，f(t))是在class c的documents中出现word w(t)的个数除以documents的总数。

multinomial model的计算：P(c)是class c的documents中word出现的次数除以word出现的总次数，P(f(t))是word w(t)出现的次数除以word出现的总次数，P(c，f(t))是在class c的documents中word w(t)出现的次数除以word出现的总次数。

(5)	Conclusions

两种models都使用Bayes框架，区别在于计算P(d|c)的方式不同，multinomial model通常比multi-variate Bernoulli model的效果好。

注：

《Text Classification from Labeled and Unlabeled Documents using EM》里使用的是multinomial model，但计算P(d|c)的方法为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/189.png)





