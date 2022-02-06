# 使用tf.GradientTape()方法计算函数y=x^2在x=3时的导数

import tensorflow as tf

x = tf.Variable(initial_value=3.)  # 将变量初始值定为3，在tf.GradientTape()的上下文内，所有计算步骤都会被记录
with tf.GradientTape() as tape:    # 定义函数
    y = tf.square(x)
y_grad = tape.gradient(y, x)       # 函数计算
print([y, y_grad])


# 矩阵计算

X=tf.constant([[1.,2.],[3.,4.]])
Y=tf.constant([[1.],[2.]])
w=tf.Variable(initial_value=[[[1.],[2.]]])
b=tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    # reduce_sum对输入张量的所有元素求和
    L=tf.reduce_sum(tf.square(tf.matmul(X,w)+b-Y))
w_grad,b_grad=tape.gradient(L,[w,b])
print(L,w_grad,b_grad)