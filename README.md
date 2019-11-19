## 一、GAN

### 1  如何用 Keras 从头搭建一维生成对抗网络

[文章地址](<https://juejin.im/post/5dcf5aba6fb9a0203161f376#heading-8>)  [github](<https://github.com/luhao2013/LearningProjects/blob/master/GAN/0%E7%94%A8%20Keras%20%E4%BB%8E%E5%A4%B4%E6%90%AD%E5%BB%BA%E4%B8%80%E7%BB%B4%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C/GAN_Keras.py>)

本文收获:

1. Keras 中 plot_model 函数可以画出模型

2. 生成器的目的是将一个已知分布映射为我们真实数据的分布训练的时候. 就是要从已知分布中生成数据, 经过生成器映射后得到的假数据的分布, 与真实样本的分布趋于一致.

3. 生成器训练的时候是整个GAN网络进行训练, 但将判别器冻结了, 假样本的标签置为1(应该是想训练生成器使得假样本趋于1,但还不太理解). 

   ```Python
   def define_gan(generator, discriminator):
   	# 将判别器的权重设为不可训练
   	discriminator.trainable = False
   ```
4. 每一个epoch, 先判别器训练, 再生成器(即整个GAN模型冻结判别器)训练. 所以生成器不需要compile.



