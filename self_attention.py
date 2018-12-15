import tensorflow as tf
import math

def check(x):
    if isinstance(x, tuple):
        x = tf.concat(x, 2)
    return x

def feed_forward(x, size, embedding, name=1):
    name = "attention_feed_forward" + str(name)
    with tf.variable_scope(name):
        w1 = tf.Variable(tf.random_normal([size, embedding], stddev=0.1), name="w1")
        b1 = tf.Variable(tf.random_normal([embedding], stddev=0.1), name="b1")
        w2 = tf.Variable(tf.random_normal([embedding, embedding], stddev=0.1), name="w2")
        b2 = tf.Variable(tf.random_normal([embedding], stddev=0.1), name="b2")
        y = tf.nn.relu(tf.tensordot(x, w1, axes=1, name="relu") + b1)
        return tf.tensordot(y, w2, axes=1, name="feed") + b2

def self_attention(inputs, seq_lengths, dropout, params, layers=1):
    """
    inputs: [batch_size, max_time, embedded_size]
    output: [batch_size, max_time, max_time]
    """

    name = "self_attention" + str(layers)
    scale = math.sqrt(params.attention_size)
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    embedded_size = inputs.shape[2].value
    attention_size = params.attention_size

    with tf.variable_scope(name):
        w_q = tf.Variable(tf.random_normal([embedded_size, attention_size], stddev=0.01), name="w_q")
        w_k = tf.Variable(tf.random_normal([embedded_size, attention_size], stddev=0.01), name="w_k")
        w_v = tf.Variable(tf.random_normal([embedded_size, attention_size], stddev=0.01), name="w_v")

        b_q = tf.Variable(tf.random_normal([attention_size], stddev=0.01), name="b_q")
        b_k = tf.Variable(tf.random_normal([attention_size], stddev=0.01), name="b_k")
        b_v = tf.Variable(tf.random_normal([attention_size], stddev=0.01), name="b_v")
        
        q = tf.nn.dropout(tf.tensordot(inputs, w_q, axes=1, name="q") + b_q, 1 - dropout)
        k = tf.nn.dropout(tf.tensordot(inputs, w_k, axes=1, name="k") + b_k, 1 - dropout)
        v = tf.nn.dropout(tf.tensordot(inputs, w_v, axes=1, name="v") + b_v, 1 - dropout)

        q_k = tf.matmul(q, k, transpose_b=True, name="q_k") # [batch_size, max_time, max_time]
        q_k = q_k / tf.constant(scale) # scaled
        q_k = tf.nn.softmax(q_k)

        output = tf.nn.dropout(tf.matmul(q_k, v, name="output"), 1 - dropout) # [batch_size, max_time, attention_size]

    return output
