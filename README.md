## 一、GAN

### 1  用 Keras 从头搭建一维生成对抗网络

[文章地址](<https://juejin.im/post/5dcf5aba6fb9a0203161f376#heading-8>)  [Github](<https://github.com/luhao2013/LearningProjects/blob/master/GAN/0%E7%94%A8%20Keras%20%E4%BB%8E%E5%A4%B4%E6%90%AD%E5%BB%BA%E4%B8%80%E7%BB%B4%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C/GAN_Keras.py>)  [知乎理论](<https://zhuanlan.zhihu.com/p/33752313>)

**本文收获:**

1. Keras 中 plot_model 函数可以画出模型

2. 生成器的目的是将一个已知分布映射为我们真实数据的分布训练的时候. 就是要从已知分布中生成数据, 经过生成器映射后得到的假数据的分布, 与真实样本的分布趋于一致.

3. 生成器训练的时候是整个GAN网络进行训练, 但将判别器冻结了, 假样本的标签置为1(应该是想训练生成器使得假样本趋于1,但还不太理解).  

   **补充理解:** 在生成时候是最小化损失函数即min, 而在判别的时候是最大化损失即max, 所以这里标签相反, 就都是max了, 是编程技巧吧. 注意GAN的损失函数是交叉熵的相反数, 否则应该都是min.

   ```Python
   def define_gan(generator, discriminator):
   	# 将判别器的权重设为不可训练
   	discriminator.trainable = False
   ```

4. 每一个epoch, 先判别器训练, 再生成器(即整个GAN模型冻结判别器)训练. 所以生成器不需要compile.

### 2 用tf1实现gan_mlp生成数字
[GitHub](https://github.com/luhao2013/LearningProjects/blob/master/GAN/1%E7%94%A8tf1.0%E7%94%9F%E6%88%90%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97)

1. jupyter文件中，有详细的注释，使用MLP做生成器判别器
2. 先构建的静态图，再运行
3. py文件中将其**重构为一个类**，实现一种tensorflow代码风格

## 二、word2vec
### 1. word2vec_tf
参考 [1](https://mp.weixin.qq.com/s?__biz=MzAxMjMwODMyMQ==&mid=2456341414&idx=1&sn=37949abb78e24f8fe91d999ee734c9fc&chksm=8c2fb1a8bb5838be4cba815b933d40bd7f4dfc5a4d242e3ca75efc95301758c99fc10a778d58&mpshare=1&scene=1&srcid=&sharer_sharetime=1575302374751&sharer_shareid=5d0dc67e21d6f0fd3facbc8bdea36a45#rd) [2](https://blog.csdn.net/xiaosongshine/article/details/84968883)
 [知乎](https://zhuanlan.zhihu.com/p/75116943) [tensorflow tutorial](https://github.com/tensorflow/tensorflow/blob/e44f32560dc3cc340fa3e9ab4a0ea2268936d179/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)

## 三、图像
### 1. mnist_tf2
[code来源](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/neural_network.ipynb)
[1](https://mp.weixin.qq.com/s/NHqv0LO3yn1d2RJWUB6T6g)

## 四、AutoEncoder

### 1.simple_autoencoder

[原文地址](<https://towardsdatascience.com/machine-learning-autoencoders-712337a07c71>)  [GitHub](<https://github.com/luhao2013/LearningProjects/tree/master/AutoEncoder/0simple_autoencoder_tf2>)

**本文收获：**

1. tf.keras.utils.plot_model()可以画出模型。
2. tf2参数重用很简单， tf.keras.Model(input, output)，定义两个model，就可以参数重用。

