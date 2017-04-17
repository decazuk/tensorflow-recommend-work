from collections import deque
from six import next
import readers

import tensorflow as tf
import numpy as np

import time

np.random.seed(42)

u_num = 122979
i_num = 210420

batch_size = 1000
dims = 15
max_epochs = 150

place_device = "/cpu:0"

def get_data():
    df = readers.read_file('./data/work_experiences.dat', sep="::")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test

def clip(x):
    return np.clip(x, 1.0, 120.0)

def model(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0"):
    with tf.device("/cpu:0"):
        bias_global = tf.get_variable("bias_global", shape=[])

        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])

        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")

        w_user = tf.get_variable("embd_user", shape=[user_num, dim],
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable("embd_item", shape=[item_num, dim],
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")

    with tf.device(device):
        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")
        regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")

    return infer, regularizer

def loss(infer, regularizer, work_time_batch, learning_rate=0.1, reg=0.1, device="/cpu:0"):
    with tf.device(device):
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, work_time_batch))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))

        train_op = tf.train.FtrlOptimizer(learning_rate).minimize(cost)
    return cost, train_op

df_train, df_test = get_data()

samples_per_batch = len(df_train) // batch_size
print("Number of train samples %d, test samples %d, samples per batch %d" % 
      (len(df_train), len(df_test), samples_per_batch))

# Peeking at the top 5 user values
print(df_train["user"].head()) 
print(df_test["user"].head())

# Peeking at the top 5 item values
print(df_train["item"].head())
print(df_test["item"].head())

# Peeking at the top 5 work_time values
print(df_train["work_time"].head())
print(df_test["work_time"].head())

# Using a shuffle iterator to generate random batches, for training
iter_train = readers.ShuffleIterator([df_train["user"],
                                     df_train["item"],
                                     df_train["work_time"]],
                                     batch_size=batch_size)

iter_test = readers.OneEpochIterator([df_test["user"],
                                     df_test["item"],
                                     df_test["work_time"]],
                                     batch_size=-1)

user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
work_time_batch = tf.placeholder(tf.float32, shape=[None])

infer, regularizer = model(user_batch, item_batch, user_num=u_num, item_num=i_num, dim=dims, device=place_device)
_, train_op = loss(infer, regularizer, work_time_batch, learning_rate=0.10, reg=0.05, device=place_device)

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print("%s\t%s\t%s\t%s" % ("Epoch", "Train Error", "Val Error", "Elapsed Time"))
    errors = deque(maxlen=samples_per_batch)
    start = time.time()
    for i in range(max_epochs * samples_per_batch):
        users, items, work_times = next(iter_train)
        _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                              item_batch: items,
                                                              work_time_batch: work_times})
        pred_batch = clip(pred_batch)
        errors.append(np.power(pred_batch - work_times, 2))
        if i % samples_per_batch == 0:
            train_err = np.sqrt(np.nanmean(errors))
            test_err2 = np.array([])
            for users, items, work_times in iter_test:
                pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                        item_batch: items})
                pred_batch = clip(pred_batch)
                test_err2 = np.append(test_err2, np.power(pred_batch - work_times, 2))
            end = time.time()

            print("%02d\t%.3f\t\t%.3f\t\t%.3f secs" % (i // samples_per_batch, train_err, np.sqrt(np.nanmean(test_err2)), end - start))
            start = end
    saver.save(sess, './save/')

