import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import matplotlib.pyplot as plt
import random


# from rnn_cell import NLSTMCell
# inputs=tf.placeholder(tf.float32,[None,20])
# inputs2=tf.placeholder(tf.float32,[None,20])

# cell=NLSTMCell(num_units=4,depth=2,state_is_tuple=True)
# init_state = (cell.zero_state(batch_size=2,dtype=tf.float32),cell.zero_state(batch_size=2,dtype=tf.float32),cell.zero_state(batch_size=2,dtype=tf.float32))
# print(cell.zero_state(batch_size=2,dtype=tf.float32))
#
# # output,new_state=cell(inputs,state=init_state)
# input1=np.random.randn(2*20)
# input1=np.reshape(input1,(-1,20))
# input2=np.random.randn(2*20)
# input2=np.reshape(input2,(-1,20))
# output1,new_state=cell(inputs,state=init_state)
#
# output,new_state=cell(inputs2,state=new_state)
# sess=tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run([new_state],feed_dict={inputs:input1,inputs2:input2}))


def attention(inputs,attention_size,time_major=False,return_alphas=True):
    if isinstance(inputs,tuple):
        inputs = tf.concat(inputs,2)
    if time_major:
        inputs = tf.array_ops.transpose(inputs,[1,0,2])
    hidden_size = inputs.shape[2].value
    print('hidden_size',hidden_size)

    w_omega = tf.Variable(tf.random_normal([hidden_size,attention_size],stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size],stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size,1],stddev=0.1))

    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(inputs,w_omega,axes=1)+b_omega)
        print('v:',v)
        print('v:',v.get_shape())

    vu=tf.tensordot(v,u_omega,name='vu',axes=1)
    print('vu:',vu.get_shape())
    alphas = tf.nn.softmax(vu,name='alphas')
    print('alphas:',alphas.get_shape())
    print('inputs:',inputs.get_shape())


    output= tf.reduce_mean(inputs * tf.expand_dims(alphas,-1),1)
    print('output:',output.get_shape())

    if not return_alphas:
        return output
    else:
        return output,alphas