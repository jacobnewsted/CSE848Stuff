import tensorflow as tf
import numpy as np
import random
import cv2
import os

def sample_data():
    data = []
    temp = np.genfromtxt('Map1.csv', delimiter=',')
    for i in range(len(temp)):
        for j in range(len(temp[0])):
            data.append([i * len(temp[0]) + j, temp[i][j]])
    return data

def sample_noise(m,n):
    return np.random.uniform(-1., 1., size=[m,n])

def generator(Z, hsize=[16,16], reuse=False):
    with tf.variable_scope("GAN/Generator", reuse=reuse,dtype=tf.float16):
        h1 = tf.layers.dense(Z, hsize[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2, 2)
    return out

def discriminator(X, hsize=[16,16], reuse=False):
    with tf.variable_scope("GAN/Discriminator",reuse=reuse,dtype=tf.float16):
        h1 = tf.layers.dense(X,hsize[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2, 2)
        out = tf.layers.dense(h3, 1)
    return out,h3


batch_size = 256
# 2D data with no specific shape represented by int16's
X = tf.placeholder(tf.float32,[None,2])
Z = tf.placeholder(tf.float32,[None,2])

# Generate noise
G_sample = generator(Z)
r_logits, r_rep = discriminator(X) # Generate probabilities that the sample is real
f_logits, g_rep = discriminator(G_sample, reuse=True) # Generate probabilities the sample is false from noise

# Yikes, we are calculating the total loss by summing the probability of it being real (mapped onto the tensor shape for the true logit only considering chances of it being real)
# by the probability of it being false (mapped onto the tensor shape for the fake logit considering only chances that it is fake)
disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = f_logits, labels=tf.ones_like(f_logits)))
# Getting our optimizer for back-propogation of error
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

# Most implementations use Adam but the example I am going off of uses a rectified one, doesn't matter much for this though
gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list=gen_vars)
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list=disc_vars)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

n_dsteps = 10
n_gsteps = 10
X_batch = []
Z_batch = []
for i in range(101):
    X_batch = sample_data()
    Z_batch = sample_noise(batch_size, 2)
    for j in range(n_dsteps):
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
    rrep_dstep, grep_dstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})
    for j in range(n_gsteps):
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={X: X_batch, Z: Z_batch})
    rrep_gstep, grep_gstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

    print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i,dloss,gloss))
    if i%10 == 0:
        print("%d,%f,%f\n"%(i,dloss,gloss))

result = sess.run(G_sample, feed_dict={Z: Z_batch})
print("\nGenerated Data\n")
for i in range(len(result)):
    if i%16 == 0:
        print("")
    else:
        print(int(round(result[i][1])), sep=" ", end="", flush=True)

print("\nReal Data\n")
for i in range(len(sample_data())):
    if i%16 == 0:
        print("")
    else:
        print(int(round(sample_data()[i][1])), sep=" ", end="", flush=True)
print("")