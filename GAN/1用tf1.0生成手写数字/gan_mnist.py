# coding: utf-8
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt


class GAN_MLP(object):
    """
    使用MLP做为判别器生成器，生成手写字母
    """
    def __init__(self):
        # 定义参数
        # 真实图像的size
        # img_size = X_train[0].shape[0]
        self.img_size = 784
        # 传入给generator的噪声size
        self.noise_size = 100
        # 生成器隐层参数
        self.g_units = 128
        # 判别器隐层参数
        self.d_units = 128
        # leaky ReLU的参数
        self.alpha = 0.01
        # learning_rate
        self.learning_rate = 0.001
        # label smoothing
        self.smooth = 0.1
        
        self.losses = None
        
        self.real_img = None 
        self.noise_img = None
        
        self.d_loss_real = None
        self.d_loss_fake = None
        
        self.d_loss = None
        self.g_loss = None
        
        self.d_train_opt = None
        self.g_train_opt = None
        
        self.g_vars = None
        self.d_vars = None
        
        self.build(self.img_size, self.noise_size, self.g_units, self.d_units, self.smooth)
    
    def build(self, img_size, noise_size, g_units, d_units, smooth):
        tf.reset_default_graph()

        self.real_img, self.noise_img = self.get_inputs(img_size, noise_size)

        # generator
        g_logits, g_outputs = self.get_generator(self.noise_img, g_units, img_size)

        # discriminator
        d_logits_real, d_outputs_real = self.get_discriminator(self.real_img, d_units)
        d_logits_fake, d_outputs_fake = self.get_discriminator(g_outputs, d_units, reuse=True)
        
        # Build metrics
        self.loss(d_logits_real, d_logits_fake, smooth)
        self.optimizer()
    
    
    def get_inputs(self, real_size, noise_size):
        """
        真实图像tensor与噪声图像tensor
        """
        real_img = tf.placeholder(tf.float32, [None, real_size], name='real_img')
        noise_img = tf.placeholder(tf.float32, [None, noise_size], name='noise_img')

        return real_img, noise_img


    def get_generator(self, noise_img, n_units, out_dim, reuse=False, alpha=0.01):
        """
        生成器

        noise_img: 生成器的输入
        n_units: 隐层单元个数
        out_dim: 生成器输出tensor的size，这里应该为32*32=784
        alpha: leaky ReLU系数
        """
        with tf.variable_scope("generator", reuse=reuse):
            # hidden layer
            hidden1 = tf.layers.dense(noise_img, n_units)
            # leaky ReLU
            hidden1 = tf.maximum(alpha * hidden1, hidden1)
            # dropout
            hidden1 = tf.layers.dropout(hidden1, rate=0.2)

            # logits & outputs
            logits = tf.layers.dense(hidden1, out_dim)
            outputs = tf.tanh(logits)

            return logits, outputs


    def get_discriminator(self, img, n_units, reuse=False, alpha=0.01):
        """
        判别器

        n_units: 隐层结点数量
        alpha: Leaky ReLU系数
        """

        with tf.variable_scope("discriminator", reuse=reuse):
            # hidden layer
            hidden1 = tf.layers.dense(img, n_units)
            hidden1 = tf.maximum(alpha * hidden1, hidden1)

            # logits & outputs
            logits = tf.layers.dense(hidden1, 1)
            outputs = tf.sigmoid(logits)

            return logits, outputs
        
    
    def loss(self, d_logits_real, d_logits_fake, smooth):
        # discriminator的loss
        # 识别真实图片
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, 
                                                                             labels=tf.ones_like(d_logits_real)) * (1 - smooth))
        # 识别生成的图片
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                                             labels=tf.zeros_like(d_logits_fake)))
        # 总体loss
        self.d_loss = tf.add(self.d_loss_real, self.d_loss_fake)

        # generator的loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                        labels=tf.ones_like(d_logits_fake)) * (1 - smooth))
        
    
    def optimizer(self):
        train_vars = tf.trainable_variables()

        # generator中的tensor
        self.g_vars = [var for var in train_vars if var.name.startswith("generator")]
        # discriminator中的tensor
        self.d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

        # optimizer
        self.d_train_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        self.g_train_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss, var_list=self.g_vars)
        
        
    def train(self):
        # batch_size
        batch_size = 64
        # 训练迭代轮数
        epochs = 20
        # 抽取样本数
        n_sample = 25

        # 存储测试样例
        samples = []
        # 存储loss
        losses = []
        # 保存生成器变量
        saver = tf.train.Saver(var_list = self.g_vars)
        # 开始训练
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(epochs):
                for step in range(X_train.shape[0]//batch_size):
                    batch_index = step * batch_size % X_train.shape[0]
                    batch_index = min(batch_index, X_train.shape[0] - batch_size)
                    batch = X_train[batch_index:(batch_index + batch_size)]

        #             batch = mnist.train.next_batch(batch_size)

        #             batch_images = batch[0].reshape((batch_size, 784))
                    batch_images = batch.reshape((batch_size, 784))
                    # 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
        #             batch_images = batch_images*2 - 1
                    batch_images = batch_images/255.0 * 2 - 1

                    # generator的输入噪声
                    batch_noise = np.random.uniform(-1, 1, size=(batch_size, self.noise_size))

                    # Run optimizers
                    _ = sess.run(self.d_train_opt, feed_dict={self.real_img: batch_images, self.noise_img: batch_noise})
                    _ = sess.run(self.g_train_opt, feed_dict={self.noise_img: batch_noise})

                # 每一轮结束计算loss
                train_loss_d = sess.run(self.d_loss, 
                                        feed_dict = {self.real_img: batch_images, 
                                                     self.noise_img: batch_noise})
                # real img loss
                train_loss_d_real = sess.run(self.d_loss_real, 
                                             feed_dict = {self.real_img: batch_images, 
                                                         self.noise_img: batch_noise})
                # fake img loss
                train_loss_d_fake = sess.run(self.d_loss_fake, 
                                            feed_dict = {self.real_img: batch_images, 
                                                         self.noise_img: batch_noise})
                # generator loss
                train_loss_g = sess.run(self.g_loss, 
                                        feed_dict = {self.noise_img: batch_noise})


                print("Epoch {}/{}...".format(e+1, epochs),
                      "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
                      "Generator Loss: {:.4f}".format(train_loss_g))    
                # 记录各类loss值
                losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))

                # 抽取样本后期进行观察
                sample_noise = np.random.uniform(-1, 1, size=(n_sample, self.noise_size))
                gen_samples = sess.run(self.get_generator(self.noise_img, self.g_units, self.img_size, reuse=True),
                                       feed_dict={self.noise_img: sample_noise})
                samples.append(gen_samples)

                # 存储checkpoints
                saver.save(sess, './checkpoints/generator.ckpt')
        
        self.losses= losses

        # 将sample的生成数据记录下来
        with open('train_samples.pkl', 'wb') as f:
            pickle.dump(samples, f)
            
    def plot_loss(self):
        """
        绘制loss曲线
        """
        fig, ax = plt.subplots(figsize=(20,7))
        losses = np.array(self.losses)
        plt.plot(losses.T[0], label='Discriminator Total Loss')
        plt.plot(losses.T[1], label='Discriminator Real Loss')
        plt.plot(losses.T[2], label='Discriminator Fake Loss')
        plt.plot(losses.T[3], label='Generator')
        plt.title("Training Losses")
        plt.legend()
        
    def gen_pictures(self):
        """
        使用生成器产生图片
        """
        # 加载我们的生成器变量
        saver = tf.train.Saver(var_list=self.g_vars)
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            sample_noise = np.random.uniform(-1, 1, size=(25, noise_size))
            gen_samples = sess.run(get_generator(self.noise_img, self.g_units, self.img_size, reuse=True),
                                   feed_dict={self.noise_img: sample_noise})


def view_samples(epoch, samples):
    """
    epoch代表第几次迭代的图像
    samples为我们的采样结果
    """
    fig, axes = plt.subplots(figsize=(7,7), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch][1]): # 这里samples[epoch][1]代表生成的图像结果，而[0]代表对应的logits
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    return fig, axes


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    img = X_train[50]
    plt.imshow(img.reshape((28, 28)), cmap='Greys_r')
    
    model = GAN_MLP()
    model.train()
    model.plot_loss()

    # 显示图像
    # Load samples from generator taken while training
    with open('train_samples.pkl', 'rb') as f:
        samples = pickle.load(f)
    
    _ = view_samples(-1, samples) # 显示最后一轮的outputs

