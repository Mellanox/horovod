import tensorflow as tf
import horovod.tensorflow as hvd
from horovod.tensorflow.mpi_ops import _allreduce
import time

hvd.init()

rank = hvd.rank()

n1 = 300
n2 = 20
n = n1*n2
##rank = 2
with tf.device("/gpu:0"):
  A = tf.get_variable("var",[n1,n2],dtype=tf.float32, initializer=tf.random_uniform_initializer(-10,10,rank,tf.float32))
  A2 = tf.transpose(A)
  B = tf.matmul(A,A2)
  B = tf.reshape(B,[-1])
  ##C = B

C = _allreduce(B)

var_init_op = tf.global_variables_initializer()

with tf.Session() as sess: 
  sess.run(var_init_op)  
  b,c = sess.run([B,C])

  for i in range(n):
    print("%d) %f" % (i, b[i]))
    print("Sum %d: %f" % (i, c[i]))
