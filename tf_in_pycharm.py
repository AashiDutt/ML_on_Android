# this code repository contains basics of tensorflow and linear regression model

import tensorflow as tf
#Constant nodes
#const_1 = tf.constant(value =[[1.0,2.0]],dtype = tf.float32,shape=[1,2],name = "const_1",verify_shape=True)

# create session to run a node
session = tf.Session()
#print(session.run(fetches=const_1))

# Variable nodes
# to run variable nodes use global_variables_initializer() along with session

#var_1 =tf.Variable(initial_value=[1.0],trainable=True,collections=None,validate_shape=True,caching_device=None,name='var_1',dtype=tf.float32)

#print(var_1)
#init = tf.global_variables_initializer()
#session.run(init)
#print(session.run(var_1))

# assigining new value to var_1 node is stored in a new node

#var_2 = var_1.assign([2.0])
#print(session.run(var_1))
#print(session.run(var_2))

# placeholder nodes - inputs to our models
# don't contain values until values are assigned to them

#placeholder_1 = tf.placeholder(dtype=tf.float32,shape=(1,4),name='placeholder_1')
#placeholder_2 = tf.placeholder(dtype=tf.float32,shape=(2,2),name='placeholder_2')
#print(placeholder_1)

#print(session.run(fetches=[placeholder_1,placeholder_2],feed_dict={placeholder_1:[(1.0,2.0,3.0,4.0)],placeholder_2:[(1.0,2.0),(3.0,4.0)]}))

# Operation Nodes - any node that performs some operation on the existing nodes
#const_1 = tf.constant(value=[1.0])
#const_2 = tf.constant(value=[2.0])
#placeholder_1 = tf.placeholder(dtype=tf.float32)
#results = tf.add(x = placeholder_1,y=const_2,name ='results')
#print(session.run(results,feed_dict={placeholder_1:[(2.0)]}))

# implementing y = wx+b

#w = tf.constant(value=[2.0])
#b = tf.constant(value=[1.0])
#x = tf.placeholder(dtype=tf.float32)
#y = (w * x) + b
#print(session.run(y,feed_dict={x:[5.0]}))

# loss function : actual vs expected output
# actual: output from our model after training
# expected: correct output

# optimizer : change values in model to alter losses

#x_train =[1.0,2.0,3.0,4.0]
#y_train =[2.0,3.0,4.0,5.0]
#y_actual =[1.5,2.5,3.5,4.5]
#loss = tf.reduce_sum(tf.square(y_train-y_actual))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
#train_step =optimizer.minimize(loss)


# Linear regression
# y = Wx +b
x_train =[1.0,2.0,3.0,4.0]
y_train =[-1.0,-2.0,-3.0,-34.0]
W = tf.Variable([1.0],tf.float32)  # weights
b = tf.Variable([1.0],tf.float32)  # bias
x = tf.placeholder(dtype=tf.float32) # input

y_input = tf.placeholder(tf.float32)  # feed in values during training (expected output
y_output = W*x +b                     # output

loss = tf.reduce_sum(tf.square(y_output - y_input))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = optimizer.minimize(loss)

session.run(tf.global_variables_initializer())

# prints present loss
print(session.run(fetches=loss, feed_dict={x: x_train,y_input: y_train}))

for _ in range(1000):
    session.run(train_step, feed_dict={x: x_train,y_input: y_train})   # train model for 1000 epochs

print(session.run(fetches=loss, feed_dict={x: x_train,y_input: y_train}))   # print final value of loss

print(session.run(y_output,feed_dict={x:[5.0,10.0,15.0]}))    #  giving testing input