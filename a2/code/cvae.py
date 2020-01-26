import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import zhusuan as zs
from utils import save_image_collections, shuffle

def load_mnist():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x_train, t_train = mnist.train.images, mnist.train.labels
    x_test, t_test = mnist.test.images, mnist.test.labels
    x_test = np.random.binomial(1, x_test, size=x_test.shape)
    return x_train, t_train, x_test, t_test

x_train, t_train, x_test, t_test = load_mnist()
x_dim = x_train.shape[1]
t_dim = t_train.shape[1]
z_dim = 40

@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_gen(t, x_dim, z_dim, n, n_particles=1):
    bn = zs.BayesianNet()

    # z ~ Normal(z; 0, I)
    z_mean = tf.zeros([n, z_dim])
    z = bn.normal("z", z_mean, std=1., group_ndims=1, n_samples=n_particles)

    # x_logits = f_NN(z+t)
    inputs = tf.concat(values=[z, tf.expand_dims(t, axis=0)], axis=2)
    h = tf.layers.dense(inputs, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_logits = tf.layers.dense(h, x_dim)

    # x ~ Bernoulli(x; sigmoid(x_logits))
    x_mean = bn.deterministic("x_mean", tf.sigmoid(x_logits))
    x_hat = bn.bernoulli("x", x_logits, group_ndims=1)
    return bn

@zs.reuse_variables(scope="q_net")
def build_q_net(x, t, z_dim, n_z_per_x):
    bn = zs.BayesianNet()

    # mean, std of latent z = g_NN(x+t)
    inputs = tf.concat(values=[x, t], axis=1)
    h = tf.layers.dense(tf.cast(inputs, tf.float32), 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)

    # z ~ Normal(z; z_mean, z_logstd)
    z = bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1, n_samples=n_z_per_x)
    return bn


def build_VAE():
    # Define Placeholders for Input
    n_particles = tf.placeholder(tf.int32, shape=[], name="n_particles")
    x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
    x = tf.cast(tf.less(tf.random_uniform(tf.shape(x_input)), x_input), tf.int32)
    t_input = tf.placeholder(tf.float32, shape=[None, t_dim], name="t")
    t = tf.cast(tf.less(tf.random_uniform(tf.shape(t_input)), t_input), tf.int32)
    n = tf.placeholder(tf.int32, shape=[], name="n")

    # Create VAE Model
    model = build_gen(t_input, x_dim, z_dim, n, n_particles)
    variational = build_q_net(x, tf.cast(t_input, tf.int32), z_dim, n_particles)

    # Lower Bound (ELBO)
    lower_bound = zs.variational.elbo(
        model,
        {"x": x},
        variational=variational,
        axis=0
    )
    # Apply Stochastic Gradient Variational Bayes (SGVB) that uses reparameterization
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    # Optimize
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    # Random generation
    x_gen = tf.reshape(model.observe()["x_mean"], [-1, 28, 28, 1])

    return optimizer, lower_bound, x_gen, x_input, t_input, n, n_particles


def train_VAE():
    global x_train, t_train

    # SET UP VAE COMPUTATIONAL GRAPH
    optimizer, lower_bound, x_gen, x_input, t_input, n, n_particles = build_VAE()

    # Define training/evaluation parameters
    epochs = 1000
    batch_size = 128
    iters = x_train.shape[0] // batch_size
    save_freq = 10
    test_freq = 10
    test_batch_size = 400
    test_iters = x_test.shape[0] // test_batch_size
    result_path = "results/cvae"

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            x_train, t_train = shuffle(x_train, t_train)

            lbs = []
            for i in range(iters):
                x_batch = x_train[i * batch_size:(i + 1) * batch_size]
                t_batch = t_train[i * batch_size:(i + 1) * batch_size]
                _, lb = sess.run([optimizer, lower_bound],
                                 feed_dict={x_input: x_batch,
                                            t_input: t_batch,
                                            n_particles: 1,
                                            n: batch_size})
                lbs.append(lb)
            print("Epoch {}: Lower bound = {}".format(epoch, np.mean(lbs)))

            if epoch % save_freq == 0:
                t_label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                t_label = np.broadcast_to(t_label, (100, 10)).copy()

                # Generate images for each class
                for j in range(10):
                    t_label[:, j] = 1
                    images = sess.run(x_gen,
                            feed_dict={t_input: t_label,
                                       n: 100,
                                       n_particles: 1})
                    name = os.path.join(result_path+"/label_{}".format(j),"cvae.epoch.{}.png".format(epoch))
                    save_image_collections(images, name)
                    t_label[:, j] = 0


if __name__ == '__main__':
    train_VAE()
