import tensorflow as tf
from tensorflow.python.layers import base



def maxout(inputs,k,num_units,axis=-1,name=None):


    return MaxOut(k,num_units=num_units, axis=axis, name=name)(inputs)



class MaxOut(base.Layer):

    def __init__(self,
                k,
                num_units,
                axis = -1,
                name = None,
                ** kwargs):
        super(MaxOut, self).__init__(
            name=name, trainable=True, **kwargs)
        self.axis = axis
        self.num_units = num_units
        self.k = k


    def build(self,_):

        with tf.variable_scope(self.scope_name,reuse=True):

            self.add_variable("maxout_weights",[self.num_units,self.k],dtype=tf.float32)
            self.add_variable("maxout_biases",[self.num_units,self.k],dtype=tf.float32)

        self.built = True


    def call(self, inputs, **kwargs):
        #with tf.variable_scope( reuse=True):
        w = self.add_variable("maxout_weights",None)
        b = self.add_variable("maxout_biases",None)

        z = tf.tensordot(inputs, w) + b


        z = tf.reduce_max(z, axis=1)


        return z
