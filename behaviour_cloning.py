import tensorflow as tf
import numpy as np
import os
import glob

#for machines with low vram
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

NUM_HIDDEN_LAYERS = 3
SIZE_HIDDEN_LAYERS = 64
NUM_FILTERS = 25
NUM_CONV_LAYERS = 3
CELL_SIZE = 256
    
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

    out = tf.keras.layers.Dense(size_hidden_layers)(out)

    model = tf.keras.Model(inputs, out)

    return model

def create_lstm():
    ## Parse custom network params
    size_hidden_layers = SIZE_HIDDEN_LAYERS
    cell_size = CELL_SIZE

    input_state = tf.keras.Input(
        shape=(None,size_hidden_layers), name="embeddings"
    )

    out = input_state
    # out = tf.keras.layers.Masking(mask_value=0)(out)

    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(size_hidden_layers, activation='relu')
        )(out)
    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(size_hidden_layers, activation='relu')
        )(out)

    out = tf.keras.layers.LSTM(
        cell_size, return_sequences=False, return_state=False, name="lstm"
    )(out)

    model = tf.keras.Model(input_state, out)

    return model

def create_behaviour_head():
    ## Parse custom network params
    size_hidden_layers = SIZE_HIDDEN_LAYERS
    cell_size = CELL_SIZE

    input_state = tf.keras.Input(
        shape=(cell_size), name="embeddings"
    )
    out = input_state
    out = tf.keras.layers.Dense(size_hidden_layers, activation='relu')(out)
    out = tf.keras.layers.Dense(size_hidden_layers, activation='relu')(out)
    out = tf.keras.layers.Dense(6, activation='softmax')(out)

    model = tf.keras.Model(input_state, out)

    return model

def create_behaviour(timesteps, obs_shape):
    ## Parse custom network params
    size_hidden_layers = SIZE_HIDDEN_LAYERS
    cell_size = CELL_SIZE
    num_hidden_layers = NUM_HIDDEN_LAYERS

    input_state = tf.keras.Input(
        shape=(timesteps,obs_shape[0],obs_shape[1],obs_shape[2]+8), name="observations"
    )
    out = input_state

    mask = tf.keras.layers.Masking(mask_value=0)
    mask = mask.compute_mask(tf.keras.layers.Reshape((400,-1))(out))

    conv_out = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Flatten()
            )(out)
    out = conv_out
    for i in range(num_hidden_layers):
        out = tf.keras.layers.TimeDistributed( 
            tf.keras.layers.Dense(
                units=size_hidden_layers*(num_hidden_layers-i)*2,
                activation=tf.nn.leaky_relu,
                name="fc_{0}".format(i),
            ),name="fc__{0}".format(i)
        )(out)

    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(size_hidden_layers),
        )(out)

    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(size_hidden_layers, activation='relu')
        )(inputs=out,mask=mask)
    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(size_hidden_layers, activation='relu')
        )(inputs=out,mask=mask)

    out = tf.keras.layers.LSTM(
        cell_size, return_sequences=False, return_state=False, name="lstm"
    )(inputs=out,mask=mask)
    
    out = tf.keras.layers.Dropout(0.5)(out)
    out = tf.keras.layers.Dense(size_hidden_layers, activation='relu')(out)
    out = tf.keras.layers.Dense(size_hidden_layers, activation='relu')(out)
    out = tf.keras.layers.Dense(6, activation='softmax')(out)

    model = tf.keras.Model(input_state, out)

    return model

def custom_entropy_loss(y_true, y_pred):
    l1_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    l1 = l1_loss_fn(y_true, y_pred)
    l2 = 0.0
    for prob in y_pred:
        l2 += tf.reduce_sum(prob*tf.math.log(prob))
    B = 0.1
    return l1 + (B*l2*-1)

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

def load_data(obs_data, action_data, num):
    print("Loading data...")
    x_state_train = []
    x_state_test = []
    y_train = []
    y_test = []
    
    for run in range(num):
        for timestep in range(obs_data.shape[1]-1):
            x_state = np.zeros((obs_data.shape[1],obs_data.shape[2],obs_data.shape[3],obs_data.shape[4]+8))
            for t in range(timestep+1):
                x_t = preprocess(obs_data[run][t])
                temp = x_t
                x_t[...,0] = temp[...,1]
                x_t[...,1] = temp[...,0] 
                x_t[...,2] = temp[...,6]
                x_t[...,3] = temp[...,7] 
                x_t[...,4] = temp[...,8]
                x_t[...,5] = temp[...,9] 
                x_t[...,6] = temp[...,2]
                x_t[...,7] = temp[...,3] 
                x_t[...,8] = temp[...,4]
                x_t[...,9] = temp[...,5]
                x_t[...,33] = 0
                x_state[t] = x_t            
            action = action_data[run][timestep]
            x_state_train.append(x_state)
            y_train.append((action[1]))

    # for run in range(num):
    #     for timestep in range(obs_data.shape[1]-1):
    #         x_state = np.zeros((obs_data.shape[1],obs_data.shape[2],obs_data.shape[3],obs_data.shape[4]+8))
    #         for t in range(timestep+1):
    #             x_t = preprocess(obs_data[-run][t])
    #             temp = x_t
    #             x_t[...,0] = temp[...,1]
    #             x_t[...,1] = temp[...,0] 
    #             x_t[...,2] = temp[...,6]
    #             x_t[...,3] = temp[...,7] 
    #             x_t[...,4] = temp[...,8]
    #             x_t[...,5] = temp[...,9] 
    #             x_t[...,6] = temp[...,2]
    #             x_t[...,7] = temp[...,3] 
    #             x_t[...,8] = temp[...,4]
    #             x_t[...,9] = temp[...,5]
    #             x_t[...,33] = 0
    #             x_state[t] = x_t            
    #         action = action_data[-run][timestep]
    #         x_state_train.append(x_state)
    #         y_train.append((action[1]))

    for run in range(num):
        for timestep in range(obs_data.shape[1]-1):
            x_state = np.zeros((obs_data.shape[1],obs_data.shape[2],obs_data.shape[3],obs_data.shape[4]+8))
            for t in range(timestep+1):
                x_t = preprocess(obs_data[run][t])
                x_t[...,33] = 0
                x_state[t] = x_t            
            action = action_data[run][timestep]
            x_state_train.append(x_state)
            y_train.append((action[0]))

    for run in range(15,17,1):
        for timestep in range(obs_data.shape[1]-1):
            x_state = np.zeros((obs_data.shape[1],obs_data.shape[2],obs_data.shape[3],obs_data.shape[4]+8))
            for t in range(timestep+1):
                x_t = preprocess(obs_data[run][t])
                temp = x_t
                x_t[...,0] = temp[...,1]
                x_t[...,1] = temp[...,0] 
                x_t[...,2] = temp[...,6]
                x_t[...,3] = temp[...,7] 
                x_t[...,4] = temp[...,8]
                x_t[...,5] = temp[...,9] 
                x_t[...,6] = temp[...,2]
                x_t[...,7] = temp[...,3] 
                x_t[...,8] = temp[...,4]
                x_t[...,9] = temp[...,5]
                x_t[...,33] = 0
                x_state[t] = x_t            
            action = action_data[run][timestep]
            x_state_test.append(x_state)
            y_test.append((action[1]))

    # for run in range(10,12,1):
    #     for timestep in range(obs_data.shape[1]-1):
    #         x_state = np.zeros((obs_data.shape[1],obs_data.shape[2],obs_data.shape[3],obs_data.shape[4]+8))
    #         for t in range(timestep+1):
    #             x_t = preprocess(obs_data[-run][t])
    #             temp = x_t
    #             x_t[...,0] = temp[...,1]
    #             x_t[...,1] = temp[...,0] 
    #             x_t[...,2] = temp[...,6]
    #             x_t[...,3] = temp[...,7] 
    #             x_t[...,4] = temp[...,8]
    #             x_t[...,5] = temp[...,9] 
    #             x_t[...,6] = temp[...,2]
    #             x_t[...,7] = temp[...,3] 
    #             x_t[...,8] = temp[...,4]
    #             x_t[...,9] = temp[...,5]
    #             x_t[...,33] = 0
    #             x_state[t] = x_t            
    #         action = action_data[-run][timestep]
    #         x_state_test.append(x_state)
    #         y_test.append((action[1]))

    for run in range(10,12,1):
        for timestep in range(obs_data.shape[1]-1):
            x_state = np.zeros((obs_data.shape[1],obs_data.shape[2],obs_data.shape[3],obs_data.shape[4]+8))
            for t in range(timestep+1):
                x_t = preprocess(obs_data[run][t])
                x_t[...,33] = 0
                x_state[t] = x_t            
            action = action_data[run][timestep]
            x_state_test.append(x_state)
            y_test.append((action[0]))

    x_state_train = np.array(x_state_train)
    x_state_test = np.array(x_state_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_state_train,x_state_test,y_train,y_test

def train_lstm(scenario, obs_data, x_state_train, x_state_test, y_train, y_test):
    obs_shape = obs_data.shape[2:]

    print("Loading encoder...")
    encoder = create_enc(obs_shape)
    encoder.load_weights("data/{}/pretrain/encoder".format(scenario))

    print("Loading lstm...")
    lstm = create_lstm()

    print("Loading behaviour...")
    behaviour = create_behaviour(obs_data.shape[1],obs_shape)

    e_w = encoder.get_weights()
    
    behaviour.layers[2].set_weights(e_w[0:2])
    behaviour.layers[3].set_weights(e_w[2:4])
    behaviour.layers[5].set_weights(e_w[4:6])
    behaviour.layers[7].set_weights(e_w[6:8])
    behaviour.layers[2].trainable = False
    behaviour.layers[3].trainable = False
    behaviour.layers[5].trainable = False
    behaviour.layers[7].trainable = False

    behaviour.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3),  # Optimizer
        # Loss function to minimize
        loss=[(
                custom_entropy_loss
        )],
        # List of metrics to monitor
        metrics=tf.keras.metrics.CategoricalAccuracy(),
    )

    print("Training...")
    behaviour.fit(
        x_state_train,
        y_train,
        batch_size=16,
        epochs=10,
        shuffle=True,
        validation_data=(x_state_test, y_test),
    )
    b_w = behaviour.get_weights()
    
    lstm.layers[1].set_weights(b_w[8:10])
    lstm.layers[2].set_weights(b_w[10:12])
    lstm.layers[3].set_weights(b_w[12:15])
    lstm.save_weights("data/{}/pretrain/lstm".format(scenario))

    # b_head = create_behaviour_head()
    # b_head.layers[1].set_weights(b_w[15:17])
    # b_head.layers[2].set_weights(b_w[17:19])
    # b_head.layers[3].set_weights(b_w[19:21])
    # b_head.save_weights("data/{}/behaviour/behaviour_best2".format(scenario))

def train_clones(scenario, obs_data, x_state_train, x_state_test, y_train, y_test):
    obs_shape = obs_data.shape[2:]

    print("Loading encoder...")
    encoder = create_enc(obs_shape)
    encoder.load_weights("data/{}/pretrain/encoder".format(scenario))

    print("Loading lstm...")
    lstm = create_lstm()

    print("Loading behaviour...")
    behaviour = create_behaviour(obs_data.shape[1],obs_shape)

    e_w = encoder.get_weights()
    
    behaviour.layers[2].set_weights(e_w[0:2])
    behaviour.layers[3].set_weights(e_w[2:4])
    behaviour.layers[5].set_weights(e_w[4:6])
    behaviour.layers[7].set_weights(e_w[6:8])
    behaviour.layers[2].trainable = False
    behaviour.layers[3].trainable = False
    behaviour.layers[5].trainable = False
    behaviour.layers[7].trainable = False

    lstm.load_weights("data/{}/pretrain/lstm".format(scenario))
    l_w = lstm.get_weights()
    behaviour.layers[9].set_weights(l_w[0:2])
    behaviour.layers[10].set_weights(l_w[2:4])
    behaviour.layers[11].set_weights(l_w[4:7])
    behaviour.layers[9].trainable = False
    behaviour.layers[10].trainable = False
    behaviour.layers[11].trainable = False

    behaviour.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3),  # Optimizer
        # Loss function to minimize
        loss=[(
                custom_entropy_loss
        )],
        # List of metrics to monitor
        metrics=tf.keras.metrics.CategoricalAccuracy(),
    )

    checkpoint_filepath = "data/{}/behaviour/temp/".format(scenario) + "{epoch}_{val_categorical_accuracy}"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only = True)

    print("Training...")
    behaviour.fit(
        x_state_train,
        y_train,
        batch_size=16,
        epochs=10,
        shuffle=True,
        validation_data=(x_state_test, y_test),
        callbacks=[model_checkpoint_callback]
    )

    all_acc = []
    checkpoints = glob.glob("data/{}/behaviour/temp/*.index".format(scenario))
    for checkpoint in checkpoints:
        val_acc =  float(os.path.basename(checkpoint.rstrip('.index')).split("_")[-1])
        all_acc.append(val_acc)
        # behaviour.load_weights(checkpoint.rstrip('.index'))
        # b_w = behaviour.get_weights()
        # b_head = create_behaviour_head()
        # b_head.layers[1].set_weights(b_w[15:17])
        # b_head.layers[2].set_weights(b_w[17:19])
        # b_head.layers[3].set_weights(b_w[19:21])
        # b_head.save_weights("data/{}/behaviour/behaviour_{}".format(scenario, val_acc))
    
    best_acc = np.max(all_acc)
    for checkpoint in checkpoints:
        val_acc =  float(os.path.basename(checkpoint.rstrip('.index')).split("_")[-1])
        if val_acc == best_acc:
            behaviour.load_weights(checkpoint.rstrip('.index'))
            b_w = behaviour.get_weights()
            b_head = create_behaviour_head()
            b_head.layers[1].set_weights(b_w[15:17])
            b_head.layers[2].set_weights(b_w[17:19])
            b_head.layers[3].set_weights(b_w[19:21])
            b_head.save_weights("data/{}/behaviour/behaviour_best_f".format(scenario))

def eval(scenario, obs_data, x_state_train, x_state_test, y_train, y_test):
    obs_shape = obs_data.shape[2:]

    print("Loading encoder...")
    encoder = create_enc(obs_shape)
    encoder.load_weights("data/{}/pretrain/encoder".format(scenario))

    print("Loading lstm...")
    lstm = create_lstm()

    print("Loading behaviour...")
    behaviour = create_behaviour(obs_data.shape[1],obs_shape)

    b_head = create_behaviour_head()
    b_head.load_weights("data/{}/behaviour/behaviour_best".format(scenario))
    
    e_w = encoder.get_weights()
    behaviour.layers[2].set_weights(e_w[0:2])
    behaviour.layers[3].set_weights(e_w[2:4])
    behaviour.layers[5].set_weights(e_w[4:6])
    behaviour.layers[7].set_weights(e_w[6:8])

    lstm.load_weights("data/{}/pretrain/lstm".format(scenario))
    l_w = lstm.get_weights()
    behaviour.layers[9].set_weights(l_w[0:2])
    behaviour.layers[10].set_weights(l_w[2:4])
    behaviour.layers[11].set_weights(l_w[4:7])

    b_head.load_weights("data/{}/behaviour/behaviour_best".format(scenario))
    b_w = b_head.get_weights()
    behaviour.layers[13].set_weights(b_w[0:2])
    behaviour.layers[14].set_weights(b_w[2:4])
    behaviour.layers[15].set_weights(b_w[4:6])

    behaviour.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),  # Optimizer
        # Loss function to minimize
        loss=[(
                custom_entropy_loss
        )],
        # List of metrics to monitor
        metrics=tf.keras.metrics.CategoricalAccuracy(),
    )

    results = behaviour.evaluate(x_state_test,y_test,batch_size=16)
    print("test loss, test acc:", results)

def main(scenario):
    obs_data = np.load("data/{}/training_data.npy".format(scenario))
    action_data = np.load("data/{}/action_training_data.npy".format(scenario))
    obs_data_r = np.load("data/{}/training_data_r.npy".format(scenario))
    action_data_r = np.load("data/{}/action_training_data_r.npy".format(scenario))

    obs_data_concat = np.concatenate((obs_data,obs_data_r))
    action_data_concat = np.concatenate((action_data,action_data_r))

    x_state_train, x_state_test, y_train, y_test = load_data(obs_data_concat, action_data_concat,1)
    train_lstm(scenario, obs_data, x_state_train, x_state_test, y_train, y_test)

    # obs_data = np.load("data/{}/training_data_f.npy".format(scenario))
    # action_data = np.load("data/{}/action_training_data_f.npy".format(scenario))

    x_state_train, x_state_test, y_train, y_test = load_data(obs_data, action_data,15)
    train_clones(scenario, obs_data, x_state_train, x_state_test, y_train, y_test)

    eval(scenario, obs_data, x_state_train, x_state_test, y_train, y_test)

if __name__ == "__main__":
    scenario = "CounterCircuit"
    main(scenario)