import tensorflow as tf

def ctc_loss(labels, predictions):
  batch_len = tf.cast(tf.shape(labels)[0], dtype="int64")
  input_len = tf.cast(tf.shape(predictions)[1], dtype="int64")
  label_len = tf.cast(tf.shape(labels)[1], dtype="int64")
  
  input_len = input_len * tf.ones(shape=(1, batch_len), dtype="int64")
  label_len = label_len * tf.ones(shape=(batch_len, 1), dtype="int64")

  loss = tf.keras.backend.ctc_batch_cost(labels, predictions, input_len, label_len)

  return loss