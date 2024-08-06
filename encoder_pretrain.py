import tensorflow as tf
import numpy as np
import os

NUM_HIDDEN_LAYERS = 3
SIZE_HIDDEN_LAYERS = 64
NUM_FILTERS = 25
NUM_CONV_LAYERS = 3

CELL_SIZE = 256

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
def preprocess(obs):        
    ph = np.zeros((obs.shape[0],obs.shape[1],obs.shape[2]+8))
    for layer in range(16):
        ph[...,layer] = obs[...,layer]
    for layer in range(16,19,1):
        if layer == 16:
            ph[...,layer+int(np.max(obs[...,layer]))] = np.where(obs[...,layer]>0,1,0)
        if layer == 17:
            ph[...,layer+4] = obs[...,layer]
        if layer == 18:
            ph[...,layer+int(np.max(obs[...,layer]))+4] = np.where(obs[...,layer]>0,1,0)
    for layer in range(19,26,1):
        if layer==20:
            temp = np.zeros(obs.shape[0]*obs.shape[1])
            timer = np.max(obs[...,layer])
            temp[timer] = 1
            temp = np.reshape(temp, (obs.shape[0],obs.shape[1]))
            ph[...,layer+8] = temp
        else:
            ph[...,layer+8] = obs[...,layer]         

    return ph

def create_enc(obs_shape):

    ## Parse custom network params
    num_hidden_layers = NUM_HIDDEN_LAYERS
    size_hidden_layers = SIZE_HIDDEN_LAYERS

    ## Create graph of custom network. It will under a shared tf scope such that all agents
    ## use the same model
    inputs = tf.keras.Input(
        shape=(obs_shape[0],obs_shape[1],obs_shape[2]+8), name="observations"
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

    # mu = tf.keras.layers.Dense(size_hidden_layers)(out)
    # logvar = tf.keras.layers.Dense(size_hidden_layers)(out)
    # z = Sampling()([mu,logvar])

    # model = tf.keras.Model(inputs, [mu, logvar, z])
        
    out = tf.keras.layers.Dense(size_hidden_layers)(out)
    model = tf.keras.Model(inputs, out)
    return model

def create_dy():
    ## Parse custom network params
    size_hidden_layers = SIZE_HIDDEN_LAYERS

    input_state = tf.keras.Input(
        shape=(size_hidden_layers), name="embeddings"
    )
    input_p1 = tf.keras.Input(
        shape=(6), name="p1 action"
    )
    input_p2 = tf.keras.Input(
        shape=(6), name="p2 action"
    )

    out = input_state
    
    action = tf.keras.layers.Concatenate()([input_p1,input_p2])
    action = tf.keras.layers.Dense(size_hidden_layers*2, activation='relu')(action)
    action = tf.keras.layers.Dense(size_hidden_layers*4, activation='relu')(action)

    out = tf.keras.layers.Dropout(0.5)(out)
    out = tf.keras.layers.Concatenate()([out,action])
    out = tf.keras.layers.Dense(size_hidden_layers*2, activation='relu')(out)
    out = tf.keras.layers.Dense(size_hidden_layers*4, activation='relu')(out)
    out = tf.keras.layers.Dense(size_hidden_layers)(out)

    model = tf.keras.Model([input_state,input_p1,input_p2], out)

    return model

def create_dec(obs_shape):

    ## Parse custom network params
    size_hidden_layers = SIZE_HIDDEN_LAYERS

    inputs = tf.keras.Input(
        shape=(size_hidden_layers)
    )
    out = inputs   
    
    out = tf.keras.layers.Dense(size_hidden_layers*2)(out)
    out = tf.keras.layers.Dense(size_hidden_layers*4)(out)
    out = tf.keras.layers.Dense(size_hidden_layers*6)(out)
    out = tf.keras.layers.Dense(obs_shape[0]*obs_shape[1]*(obs_shape[2]+8), activation=tf.nn.sigmoid)(out)
    out = tf.keras.layers.Reshape((obs_shape[0],obs_shape[1],obs_shape[2]+8))(out)

    model = tf.keras.Model(inputs, out)

    return model    

class AE(tf.keras.Model):
    def __init__(self, encoder, decoder, dy, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.dy = dy
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        # self.kl_loss_tracker = tf.keras.metrics.Mean(
        #     name="kl_loss"
        # )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            # self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            (x,p1,p2),y = data
            h = self.encoder(x)
            h = self.dy((h,p1,p2))
            reconstruction = self.decoder(h)
            loss = tf.keras.losses.BinaryFocalCrossentropy(axis=(1, 2))
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    loss(reconstruction, y)
                )
            )
            # kl_loss = -0.5 * (1 + var - tf.square(mu) - tf.exp(var))
            # kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            # total_loss = reconstruction_loss + kl_loss*(1/2000)
            total_loss = reconstruction_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        # self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            # "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def call(self, data):
        x,p1,p2 = data
        h = self.encoder(x)
        h = self.dy((h,p1,p2))
        reconstruction = self.decoder(h)

        return reconstruction
    
    
def train(scenario, obs_data, action_data):
    obs_shape = obs_data.shape[2:]
    
    x_state_train = []
    x_p1_train = []
    x_p2_train = []
    y_train = []

    encoder = create_enc(obs_shape)
    dy = create_dy()
    decoder = create_dec(obs_shape)

    for run in range(obs_data.shape[0]):
        for timestep in range(obs_data.shape[1]-1):
            x_state = preprocess(obs_data[run][timestep])
            x_state[...,33] = 0
            y_state = preprocess(obs_data[run][timestep+1])
            y_state[...,33] = 0
            y_state = x_state.astype(int)^y_state.astype(int)
            y_state = y_state.astype(float)
            action = action_data[run][timestep]
            x_state_train.append(x_state)
            x_p1_train.append(action[0])
            x_p2_train.append(action[1])
            y_train.append((y_state))

    x_state_train = np.array(x_state_train)
    x_p1_train = np.array(x_p1_train)
    x_p2_train = np.array(x_p2_train)
    y_train = np.array(y_train)

    ae = AE(encoder, decoder, dy)
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4))
    ae.fit((x_state_train,x_p1_train,x_p2_train),y_train, epochs=50, batch_size=32, shuffle=True)

    path = "data/{}/pretrain".format(scenario)
    if not os.path.exists(path):
        os.makedirs(path)

    encoder.save_weights("data/{}/pretrain/encoder".format(scenario))

def main(scenario):
    obs_data = np.load("data/{}/training_data.npy".format(scenario))
    action_data = np.load("data/{}/action_training_data.npy".format(scenario))
    obs_data_r = np.load("data/{}/training_data_r.npy".format(scenario))
    action_data_r = np.load("data/{}/action_training_data_r.npy".format(scenario))

    obs_data_concat = np.concatenate((obs_data,obs_data_r))
    action_data_concat = np.concatenate((action_data,action_data_r))

    train(scenario, obs_data_concat, action_data_concat)

if __name__ == "__main__":
    scenario = "CoordinationRing"
    main(scenario)