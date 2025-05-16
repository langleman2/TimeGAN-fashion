# Necessary Packages
import tensorflow as tf
import numpy as np
from utils import extract_time, rnn_cell, random_generator, batch_generator


import wandb
wandb.init(project='Conditional TimeGAN')
# 실행 이름 설정
wandb.run.name = 'Loss Check'
wandb.run.save()


def timegan_conditional(ori_data, categories, parameters):
    """TimeGAN with conditional input for different categories.

    Args:
      - ori_data: original time-series data (list of arrays, each representing a category)
      - categories: list of category labels corresponding to each time-series in ori_data
      - parameters: TimeGAN network parameters

    Returns:
      - generated_data: generated time-series data
    """
    # Initialization on the Graph
    tf.reset_default_graph()

    # Category-wise normalization
    def MinMaxScaler(data):
        """Min-Max Normalizer.

        Args:
          - data: raw data

        Returns:
          - norm_data: normalized data
          - min_val: minimum values (for renormalization)
          - max_val: maximum values (for renormalization)
        """
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val

        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)

        return norm_data, min_val, max_val

    # Normalize data per category
    norm_data_list = []
    min_vals = []
    max_vals = []   #여기 하나하나 뭐 담기는지 확인할것,,,,


    print(len(ori_data))
    print(ori_data)

    for i in range(len(ori_data)):
        norm_data, min_val, max_val = MinMaxScaler(ori_data[i])
        norm_data_list.append(norm_data)
        min_vals.append(min_val)
        max_vals.append(max_val)

    #ori_data = np.concatenate(norm_data_list, axis=0)
    categories = np.array(categories)
    
    print(ori_data.shape)
    print('norm_data_list')
    #print(norm_data_list)
    print(len(norm_data_list))

    print(min_vals)
    print(max_vals)
    print(len(min_vals), len(max_vals))
    # Basic Parameters

    ori_data = np.array(ori_data)              # 수정!
    #breakpoint()
    ori_data = ori_data[:,:, np.newaxis]  
    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, max_seq_len = extract_time(ori_data)

    # One-hot encoding of categories
    num_categories = len(np.unique(categories))
    categories_one_hot = np.eye(num_categories)[categories]

    print(categories_one_hot)

    # Network Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = dim
    gamma = 1


    args = {
    "iterations": iterations,
    "batch_size": batch_size,
    "hidden_dim": hidden_dim,
    "num_layers": num_layers
    }
    wandb.config.update(args)

    # Input place holders
    X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
    Z = tf.placeholder(tf.float32, [None, max_seq_len, z_dim], name="myinput_z")
    T = tf.placeholder(tf.int32, [None], name="myinput_t")
    C = tf.placeholder(tf.float32, [None, num_categories], name="myinput_c")  # Category input

    # Embedding function
    def embedder(X, T, C):
        """Embedding network between original feature space to latent space."""
        with tf.variable_scope("embedder", reuse=tf.AUTO_REUSE):
            X_concat = tf.concat([X, tf.tile(tf.expand_dims(C, 1), [1, max_seq_len, 1])], axis=2)
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X_concat, dtype=tf.float32, sequence_length=T)
            H = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
        return H

    # Recovery function
    def recovery(H, T, C):
        """Recovery network from latent space to original space."""
        with tf.variable_scope("recovery", reuse=tf.AUTO_REUSE):
            H_concat = tf.concat([H, tf.tile(tf.expand_dims(C, 1), [1, max_seq_len, 1])], axis=2)
            r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, H_concat, dtype=tf.float32, sequence_length=T)
            X_tilde = tf.contrib.layers.fully_connected(r_outputs, dim, activation_fn=tf.nn.sigmoid)
        return X_tilde

    # Generator function
    def generator(Z, T, C):
        """Generator function: Generate time-series data in latent space."""
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            Z_concat = tf.concat([Z, tf.tile(tf.expand_dims(C, 1), [1, max_seq_len, 1])], axis=2)
            g_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            g_outputs, g_last_states = tf.nn.dynamic_rnn(g_cell, Z_concat, dtype=tf.float32, sequence_length=T)
            E = tf.contrib.layers.fully_connected(g_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
        return E

    # Supervisor function
    def supervisor(H, T, C):
        """Generate next sequence using the previous sequence."""
        with tf.variable_scope("supervisor", reuse=tf.AUTO_REUSE):
            H_concat = tf.concat([H, tf.tile(tf.expand_dims(C, 1), [1, max_seq_len, 1])], axis=2)
            s_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers - 1)])
            s_outputs, s_last_states = tf.nn.dynamic_rnn(s_cell, H_concat, dtype=tf.float32, sequence_length=T)
            S = tf.contrib.layers.fully_connected(s_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
        return S

    # Discriminator function
    def discriminator(H, T, C):
        """Discriminate the original and synthetic time-series data."""
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            H_concat = tf.concat([H, tf.tile(tf.expand_dims(C, 1), [1, max_seq_len, 1])], axis=2)
            d_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, H_concat, dtype=tf.float32, sequence_length=T)
            Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None)
            # Return both Y_hat and the intermediate feature representation
            return Y_hat, d_outputs

    # Embedder & Recovery
    H = embedder(X, T, C)
    X_tilde = recovery(H, T, C)

    # Generator
    E_hat = generator(Z, T, C)
    H_hat = supervisor(E_hat, T, C)
    H_hat_supervise = supervisor(H, T, C)

    # Synthetic data
    X_hat = recovery(H_hat, T, C)

    # Discriminator outputs with intermediate features
    Y_fake, D_fake_features = discriminator(H_hat, T, C)
    Y_real, D_real_features = discriminator(H, T, C)
    Y_fake_e, D_fake_e_features = discriminator(E_hat, T, C)

    # Feature matching loss (L2 loss between real and fake features)
    feature_matching_loss = tf.reduce_mean(tf.square(tf.reduce_mean(D_real_features, axis=0) - tf.reduce_mean(D_fake_features, axis=0)))

    # Variables
    e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
    r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
    g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
    d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

    # Losses
    D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
    G_loss_S = tf.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])
    G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X, [0])[1] + 1e-6)))
    G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat, [0])[0]) - (tf.nn.moments(X, [0])[0])))
    G_loss_V = G_loss_V1 + G_loss_V2
    # Feature Matching Loss의 가중치
    lambda_fm = 100
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V + lambda_fm * feature_matching_loss

    E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10 * tf.sqrt(E_loss_T0)
    E_loss = E_loss0 + 0.1 * G_loss_S

    # Optimizers
    E0_solver = tf.train.AdamOptimizer().minimize(E_loss0, var_list=e_vars + r_vars)
    E_solver = tf.train.AdamOptimizer().minimize(E_loss, var_list=e_vars + r_vars)
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=g_vars + s_vars)
    GS_solver = tf.train.AdamOptimizer().minimize(G_loss_S, var_list=g_vars + s_vars)

    # Training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Embedding network training
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb, C: categories_one_hot[:batch_size]})

        wandb.log({"Embedding loss": step_e_loss})
        if itt % 1000 == 0:
            print('step: '+str(itt)+'/'+str(iterations)+', e_loss: '+str(np.round(np.sqrt(step_e_loss),4)))
             

    print('Finsh Embedding network')

    # Training the combined networks
    for itt in range(iterations):
        # Generator training (twice more than discriminator training)
        for _ in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V], 
                                                                       feed_dict={Z: Z_mb, X: X_mb, T: T_mb, C: categories_one_hot[:batch_size]})
            _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb, C: categories_one_hot[:batch_size]})

        # Discriminator training
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb, C: categories_one_hot[:batch_size]})


        wandb.log({"g_loss_u loss": step_g_loss_u})
        wandb.log({"g_loss_s loss": step_g_loss_s})
        wandb.log({"g_loss_v loss": step_g_loss_v})
        wandb.log({"d_loss loss": step_d_loss})


        # Print multiple checkpoints
        if itt % 1000 == 0:
            print('step: '+str(itt)+'/'+str(iterations)+', d_loss: '+str(np.round(step_d_loss,4))+
                  ', g_loss_u: '+str(np.round(step_g_loss_u,4))+', g_loss_s: '+str(np.round(np.sqrt(step_g_loss_s),4)) + 
                  ', g_loss_v: '+str(np.round(step_g_loss_v,4)))

    ## Final Outputs
    # Generate data
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, T: ori_time, C: categories_one_hot})


#    print('생성된 애들 체크')
#    print(generated_data_curr)
#    print(generated_data_curr[0])
#    print(len(generated_data_curr))


 #정규화 코드 고쳐야돼   
    # Renormalization per category
    generated_data = []
    start = 0
    print(len(norm_data_list))
    for i in range(len(norm_data_list)):  #norm_data_list 잘못됨!!!
        
        generated_data_cat = generated_data_curr[i]
        generated_data_cat = generated_data_cat * (max_vals[i] + 1e-7) + min_vals[i]
        #print(generated_data_cat)
        #print(len(generated_data_cat))
        generated_data.append(generated_data_cat)
        

    return generated_data