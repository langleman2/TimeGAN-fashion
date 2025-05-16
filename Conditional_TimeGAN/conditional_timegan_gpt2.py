# Necessary Packages
import tensorflow as tf
import numpy as np
from utils import extract_time, rnn_cell, random_generator, batch_generator

import wandb
wandb.init(project='Conditional TimeGAN')
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
    def MinMaxScaler(data, split_size=12):
        """Min-Max Normalizer.
        Args:
          - data: raw data (num_samples, seq_len, dim)
          - split_size: size of each group for normalization
        Returns:
          - norm_data: normalized data
          - min_vals: list of min values for each group
          - max_vals: list of max values for each group
        """
        norm_data_list = []
        min_vals = []
        max_vals = []
        
        for i in range(0, data.shape[2], split_size):
            data_split = data[:, :, i:i+split_size]
            min_val = np.min(np.min(data_split, axis=0), axis=0)
            max_val = np.max(np.max(data_split, axis=0), axis=0)
            norm_data_split = (data_split - min_val) / (max_val - min_val + 1e-7)
            norm_data_list.append(norm_data_split)
            min_vals.append(min_val)
            max_vals.append(max_val)
        
        norm_data = np.concatenate(norm_data_list, axis=2)
        return norm_data, min_vals, max_vals

    def inverse_MinMaxScaler(data, min_vals, max_vals, split_size=12):
        """Inverse Min-Max Scaler.
        Args:
          - data: normalized data (num_samples, seq_len, dim)
          - min_vals: list of min values for each group
          - max_vals: list of max values for each group
          - split_size: size of each group for normalization
        Returns:
          - original_data: re-scaled data
        """
        original_data_list = []
        for i in range(0, data.shape[2], split_size):
            data_split = data[:, :, i:i+split_size]
            min_val = min_vals[i // split_size]
            max_val = max_vals[i // split_size]
            original_data_split = data_split * (max_val - min_val + 1e-7) + min_val
            original_data_list.append(original_data_split)
        
        original_data = np.concatenate(original_data_list, axis=2)
        return original_data

    # Normalize data
    norm_data, min_vals, max_vals = MinMaxScaler(ori_data, split_size=12)
    ori_data = norm_data

    # Category encoding
    categories = np.array(categories)
    num_categories = len(np.unique(categories))
    categories_one_hot = np.eye(num_categories)[categories]

    # Basic Parameters
    ori_data = np.array(ori_data)
    ori_data = ori_data[:, :, np.newaxis]
    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, max_seq_len = extract_time(ori_data)

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

    # Feature matching loss
    feature_matching_loss = tf.reduce_mean(tf.square(tf.reduce_mean(D_real_features, axis=0) - tf.reduce_mean(D_fake_features, axis=0)))

    # Losses
    D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
    G_loss_S = tf.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])
    G_loss_V = feature_matching_loss
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V

    E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10 * tf.sqrt(E_loss_T0)
    E_loss = E_loss0 + 0.1 * G_loss_S

    # Solver
    E_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
    R_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
    G_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    S_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
    D_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

    E_solver = tf.train.AdamOptimizer().minimize(E_loss, var_list=E_vars + R_vars)
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=D_vars)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=G_vars + S_vars)

    # Sessions
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Training
    for itt in range(iterations):
        # Batch generator for original data
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

        # Batch generator for category
        idx = np.random.permutation(len(ori_data))[:batch_size]
        C_mb = categories_one_hot[idx]

        # Train embedder
        _, step_e_loss = sess.run([E_solver, E_loss], feed_dict={X: X_mb, T: T_mb, C: C_mb})

        # Train generator and supervisor
        _, step_g_loss_s = sess.run([G_solver, G_loss_S], feed_dict={X: X_mb, T: T_mb, C: C_mb, Z: Z_mb})
        _, step_g_loss_u = sess.run([G_solver, G_loss_U], feed_dict={X: X_mb, T: T_mb, C: C_mb, Z: Z_mb})

        # Train discriminator
        _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, C: C_mb, Z: Z_mb})

        # Train overall generator
        _, step_g_loss_v = sess.run([G_solver, G_loss_V], feed_dict={X: X_mb, T: T_mb, C: C_mb, Z: Z_mb})

        if itt % 100 == 0:
            print('Iteration: ' + str(itt) + '/' + str(iterations))
            print('Step E Loss: ' + str(np.round(np.sqrt(step_e_loss), 4)) + 
                  ', Step G Loss U: ' + str(np.round(np.sqrt(step_g_loss_u), 4)) + 
                  ', Step G Loss S: ' + str(np.round(np.sqrt(step_g_loss_s), 4)) + 
                  ', Step D Loss: ' + str(np.round(np.sqrt(step_d_loss), 4)))
                  
            wandb.log({
                "Step E Loss": np.round(np.sqrt(step_e_loss), 4),
                "Step G Loss U": np.round(np.sqrt(step_g_loss_u), 4),
                "Step G Loss S": np.round(np.sqrt(step_g_loss_s), 4),
                "Step D Loss": np.round(np.sqrt(step_d_loss), 4),
            })

    ## Final Output
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, T: ori_time, C: categories_one_hot})

    # Renormalization
    generated_data = inverse_MinMaxScaler(generated_data_curr, min_vals, max_vals, split_size=12)

    return generated_data
