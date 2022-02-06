'''
某城市2013至2017年房价拟合线性方程
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 导入数据
X = np.array([2013,2014,2015,2016,2017],dtype=float)
Y = np.array([12000,14000,15000,16500,17500],dtype=float)
# 归一化
X = (X-X.min())/(X.max()-X.min())
Y = (Y-Y.min())/(Y.max()-Y.min())

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variable=[a,b]

num_epoch=16000
# 声明梯度下降优化器（optimizer），其学习率为1e-3，调用方法为optimizer.apply_gradients()方法
optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3)

for i in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred=a*X+b
        loss=tf.reduce_sum(tf.square(Y-y_pred))
    grads=tape.gradient(loss,variable)
    optimizer.apply_gradients(grads_and_vars=zip(grads,variable))
    if i%1000==0 :
        print("---------------------------------")
        print(i,a,b)
        print("---------------------------------")


print(a,b)
# <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.98179364> <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.0545579>
plt.plot(X,Y,"ob")
plt.plot(X,a*X+b)
plt.show()