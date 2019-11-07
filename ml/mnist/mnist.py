from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST",one_hot=True)

print(mnist.train.images.shape)
print(mnist.train.labels.shape)

#获取第二张图片
image = mnist.train.images[1,:]
#将图像数据还原成28*28的分辨率
image = image.reshape(28,28)
#打印对应的标签
print(mnist.train.labels[1])

plt.figure()
plt.imshow(image)
plt.show()

