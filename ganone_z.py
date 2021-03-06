from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf

import os
from tensorflow.keras import layers

from tensorflow import keras

# Helper libraries
import imageio
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython import display
from scipy import signal

####################################################
print(tf.__version__)

trainECG_0 = tf.random.normal([1100,400])
print(trainECG_0,'#########################')
trainECG = np.expand_dims(trainECG_0,2)
print(trainECG.shape)

####################################################
BUFFER_SIZE = 60000
BATCH_SIZE = 10
inputLength = 400

train_dataset = tf.data.Dataset.from_tensor_slices(trainECG).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print(train_dataset)

####################################################

# 构建模型
# 生成器
def build_generator():
    generator = tf.keras.Sequential()
    generator.add(layers.Dense(inputLength*BATCH_SIZE, input_shape=(inputLength,1),dtype='float32'))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 5, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 32,kernel_size = 5, padding = 'same'))
    generator.add(layers.LeakyReLU(alpha=0.2))
    generator.add(layers.Conv1D(filters = 1,kernel_size = 5, activation = 'tanh', padding = 'same'))

    return generator

# 判别器
def build_discriminator():

    discriminator = tf.keras.Sequential()
    discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same',input_shape=(inputLength,1)))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same')) 
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.MaxPooling1D(pool_size=2))
    discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same')) 
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same')) 
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.MaxPooling1D(pool_size=2))
    discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same')) 
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same')) 
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.MaxPooling1D(pool_size=2))
    discriminator.add(layers.Flatten(input_shape=(inputLength,1)))
    discriminator.add(layers.Dense(64,dtype='float32'))
    discriminator.add(layers.Dropout(0.4))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Dense(1, activation='tanh',dtype='float32'))

    return discriminator


# 实例化
generator = build_generator()
discriminator = build_discriminator()

#测试运行  通过
noise = tf.random.normal([20,400,1])

generateECG = generator(noise)

print(generateECG)

discriminator(generateECG)

# 定义优化器和损失函数
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# 判别损失 判别器要做两件事情，既要真的趋近于1，又要假的趋近于0
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(-0.9*tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(0.9*tf.ones_like(fake_output), fake_output)
    total_loss = 0.4*real_loss+0.6*fake_loss
    return total_loss

# 生成损失  生成器使得假的趋近于1
def generator_loss(fake_output):
    return cross_entropy(-0.9*tf.ones_like(fake_output), fake_output)

# 定义优化器
generator_optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
total_optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# 定义训练
EPOCHS = 20
noise_dim = inputLength
num_examples_to_generate = 5


# 我们将重复使用该种子（因此在动画 GIF 中更容易可视化进度）
# seed = tf.random.normal([num_examples_to_generate, noise_dim, 1],dtype='float32')
seed = tf.random.normal([num_examples_to_generate, noise_dim, 1],dtype='float32')


seed = tf.cos(4*seed)

# 单步训练
# 注意 `tf.function` 的使用
# 该注解使函数被“编译”
@tf.function
def train_step(ECG):
    noise = tf.random.normal([BATCH_SIZE, noise_dim,1])
    noise = tf.cos(4*noise)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape,tf.GradientTape() as total_tape:
      generated_ECG = generator(noise, training=True)

      real_output = discriminator(ECG, training=True)
      fake_output = discriminator(generated_ECG, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
      total_loss = tf.tanh(tf.abs(gen_loss)-tf.abs(disc_loss))
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_total = total_tape.gradient(total_loss, generator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    total_optimizer.apply_gradients(zip(gradients_of_total,generator.trainable_variables))
    return gen_loss,disc_loss,total_loss

# 定义训练
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        cunt = 0
        for image_batch in dataset:
            cunt+=1
            gen_loss,disc_loss,total_loss = train_step(image_batch)
            if cunt%100==0:
                # 继续进行时为 GIF 生成图像
                display.clear_output(wait=True)
                generate_and_save_images(generator, 
                                            epoch + 
                                            1,
                                            seed)
                print ('gen loss {} __ dis loss {} __tol loss{}'.format(gen_loss, disc_loss, total_loss))
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # 最后一个 epoch 结束后生成图片
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                            epochs,
                            seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig, axs = plt.subplots(5, 1)
    for i in range(5):
        plot_sigs = predictions[i].numpy()
        axs[i].plot(plot_sigs)
        axs[i].axis('off')
   
    plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()
#     plt.close()

train(train_dataset, 500)

generator.save('generator_model3.h5')

noise = tf.random.normal([BATCH_SIZE, noise_dim,1])
noise = tf.cos(4*noise)
epoch  = epoch+1
generate_and_save_images(generator, epoch, noise)

# 使用 epoch 数生成单张图片
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import IPython
if IPython.version_info > (6,2,0,''):
  display.Image(filename=anim_file)
