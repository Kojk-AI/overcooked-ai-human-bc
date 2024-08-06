import tensorflow as tf
import numpy as np
import os

NUM_HIDDEN_LAYERS = 3
SIZE_HIDDEN_LAYERS = 64
NUM_FILTERS = 25
NUM_CONV_LAYERS = 3
CELL_SIZE = 256

def _create_lstm():
    ## Parse custom network params
    size_hidden_layers = SIZE_HIDDEN_LAYERS
    cell_size = CELL_SIZE

    input_state = tf.keras.Input(
        shape=(None,size_hidden_layers), name="embeddings"
    )
    input_h = tf.keras.Input(
        shape=(cell_size), name="h"
    )
    input_c = tf.keras.Input(
        shape=(cell_size), name="c"
    )

    out = input_state
    # out = tf.keras.layers.Masking(mask_value=0)(out)

    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(size_hidden_layers, activation='relu')
        )(out)
    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(size_hidden_layers, activation='relu')
        )(out)

    lstm_out, h_out, c_out = tf.keras.layers.LSTM(
        cell_size, return_sequences=False, return_state=True, name="lstm"
    )(inputs=out, initial_state=[input_h,input_c])

    model = tf.keras.Model([input_state,input_h,input_c], [lstm_out, h_out, c_out])

    return model

def _create_behaviour_model(temperature=1.0):
    ## Parse custom network params
    size_hidden_layers = SIZE_HIDDEN_LAYERS
    cell_size = CELL_SIZE

    input_state = tf.keras.Input(
        shape=(cell_size), name="embeddings"
    )
    out = input_state
    out = tf.keras.layers.Dense(size_hidden_layers, activation='relu')(out)
    out = tf.keras.layers.Dense(size_hidden_layers, activation='relu')(out)
    out = tf.keras.layers.Dense(6)(out)
    out = Temperature(temperature)(out)

    model = tf.keras.Model(input_state, out)

    return model

def _create_enc(layout_size):

    ## Parse custom network params
    num_hidden_layers = NUM_HIDDEN_LAYERS
    size_hidden_layers = SIZE_HIDDEN_LAYERS

    inputs = tf.keras.Input(
        shape=(layout_size[0],layout_size[1],34), name="observations"
    )
    out = inputs

    # Apply dense hidden layers, if any
    conv_out = tf.keras.layers.Flatten()(out)
    out = conv_out
    for i in range(num_hidden_layers):
        out = tf.keras.layers.Dense(
                units=size_hidden_layers*(num_hidden_layers-i)*2,
                activation=tf.nn.leaky_relu,
                name="fc_{0}".format(i),
            )(out)

    out = tf.keras.layers.Dense(size_hidden_layers)(out)

    model = tf.keras.Model(inputs, out)

    return model

class Temperature(tf.keras.layers.Layer):
  def __init__(self, temperature):
    super(Temperature, self).__init__()
    self.temperature = tf.Variable( initial_value = [temperature], trainable=False)
    
  def call(self, final_output):
    return final_output/ self.temperature
  