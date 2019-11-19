# 在一个一维函数上训练一个生成对抗网络
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot

# 定义独立的判别器模型
def define_discriminator(n_inputs=2):
	model = Sequential()
	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
	model.add(Dense(1, activation='sigmoid'))
	# 编译模型
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# 定义独立的生成器模型
def define_generator(latent_dim, n_outputs=2):
	model = Sequential()
	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
	model.add(Dense(n_outputs, activation='linear'))
	return model

# 定义合并的生成器和判别器模型，来更新生成器
def define_gan(generator, discriminator):
	# 将判别器的权重设为不可训练
	discriminator.trainable = False
	# 连接它们
	model = Sequential()
	# 加入生成器
	model.add(generator)
	# 加入判别器
	model.add(discriminator)
	# 编译模型
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

# 生成 n 个真实样本和类标签
def generate_real_samples(n):
	# 生成 [-0.5, 0.5] 范围内的输入值
	X1 = rand(n) - 0.5
	# 生成输出值 X^2
	X2 = X1 * X1
	# 堆叠数组
	X1 = X1.reshape(n, 1)
	X2 = X2.reshape(n, 1)
	X = hstack((X1, X2))
	# 生成类标签
	y = ones((n, 1))
	return X, y

# 生成隐空间中的点作为生成器的输入
def generate_latent_points(latent_dim, n):
	# 在隐空间中生成点
	x_input = randn(latent_dim * n)
	# 为网络调整一个 batch 输入的维度大小
	x_input = x_input.reshape(n, latent_dim)
	return x_input

# 用生成器生成 n 个假样本和类标签
def generate_fake_samples(generator, latent_dim, n):
	# 在隐空间中生成点
	x_input = generate_latent_points(latent_dim, n)
	# 预测输出值
	X = generator.predict(x_input)
	# 创建类标签
	y = zeros((n, 1))
	return X, y

# 评估判别器并且绘制真假点
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
	# 准备真实样本
	x_real, y_real = generate_real_samples(n)
	# 在真实样本上评估判别器
	_, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
	# 准备假样本
	x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
	# 在假样本上评估判别器
	_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
	# 总结判别器性能
	print(epoch, acc_real, acc_fake)
	# 绘制真假数据的散点图
	pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
	pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
	pyplot.show()

# 训练生成器和判别器
def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000):
	# 用一半的 batch 数量来训练判别器
	half_batch = int(n_batch / 2)
	# 手动遍历 epoch
	for i in range(n_epochs):
		# 准备真实样本
		x_real, y_real = generate_real_samples(half_batch)
		# 准备假样本
		x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# 更新判别器
		d_model.train_on_batch(x_real, y_real)
		d_model.train_on_batch(x_fake, y_fake)
		# 在隐空间中准备点作为生成器的输入
		x_gan = generate_latent_points(latent_dim, n_batch)
		# 为假样本创建反标签  ## notice：这里的标签是1
		y_gan = ones((n_batch, 1))
		# 通过判别器的误差更新生成器
		gan_model.train_on_batch(x_gan, y_gan)
		# 为每 n_eval epoch 模型做评估
		if (i+1) % n_eval == 0:
			summarize_performance(i, g_model, d_model, latent_dim)

# 隐空间的维度
latent_dim = 5
# 创建判别器
discriminator = define_discriminator()
# 创建生成器
generator = define_generator(latent_dim)
# 创建生成对抗网络
gan_model = define_gan(generator, discriminator)
# 训练模型
train(generator, discriminator, gan_model, latent_dim)
