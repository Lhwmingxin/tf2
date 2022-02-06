import tensorflow as tf

# 定义一个随机数
random_float = tf.random.uniform(shape=())

# 定义一个有两个元素的零向量
zero_vector = tf.zeros(shape=(2))

# 定义两个2*2的常量矩阵
A = tf.constant([[2., 2.], [3., 3.]])
B = tf.constant([[5, 6], [7, 8]])
print(A)
print(A.shape)
print(A.dtype)
print(A.numpy())
