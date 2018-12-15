import tensorflow as tf


def attention(inputs, dropout, params, task=0, layers=1, class_size=3):
    """
    Input: [batch_size, max_time, hidden_size]
    """
    name = "attention" + str(layers)
    
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
    # hidden_size = params.attention_size
    attention_size = params.attention_size

    with tf.variable_scope(name):
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        v = tf.nn.dropout(tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega), 1 - dropout) # [batch_size, max_time, attention_size]
        if task==0:
            u_omega = tf.Variable(tf.random_normal([attention_size, class_size], stddev=0.1))
            vu = tf.nn.dropout(tf.tensordot(v, u_omega, axes=1, name='quality_vu'), 1 - dropout)  # [batch_size, max_time, class_size]
            alphas = tf.nn.softmax(vu, axis=1, name='quality_alphas') # [batch_size, max_time, class_size] 
            output = tf.nn.dropout(tf.matmul(alphas, inputs, transpose_a=True), 1 - dropout) # [batch_size, class_size, hidden_size]
        else: # task==1 :=> nugget task
            u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
            vu = tf.tensordot(v, u_omega, axes=1, name='nugget_vu') 
            alphas = tf.nn.softmax(vu, name='nugget_alphas')
            output = tf.nn.dropout(inputs * tf.expand_dims(alphas, -1), 1 - dropout)

    return output
