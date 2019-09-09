import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn.model_selection import KFold
import os

INPUT_DIM = 3000
OUTPUT_DIM = 4
HIDDEN_UNITS_1 = 100
HIDDEN_UNITS_2 = 40
LEARNING_RATE = 0.1
EPOCHS = 2200
KEEP_PROB = 0.8
LOG_DIR = 'logs'

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train',  categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',  categories=categories)

num_train = len(newsgroups_train.data)
num_test  = len(newsgroups_test.data)

# max_features is an important parameter. You should adjust it.
vectorizer = TfidfVectorizer(max_features=INPUT_DIM)

X = vectorizer.fit_transform( newsgroups_train.data + newsgroups_test.data )
X_train = X[0:num_train, :]
X_test = X[num_train:num_train+num_test,:]

Y_train = newsgroups_train.target
Y_test = newsgroups_test.target

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# tf graph
sess = tf.InteractiveSession()
with tf.name_scope('input'):
    x = tf.placeholder(tf.float64, [None, INPUT_DIM], name='x_input')
    y = tf.placeholder(tf.int64, [None], name='y_input')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial, dtype=tf.float64)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial, dtype=tf.float64)

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations

# multi_layers
hidden1 = nn_layer(x, INPUT_DIM, HIDDEN_UNITS_1, 'layer1')

keep_prob = tf.placeholder(tf.float64)

with tf.name_scope('dropout_1'):
    dropped_1 = tf.nn.dropout(hidden1, keep_prob)

hidden2 = nn_layer(dropped_1, HIDDEN_UNITS_1, HIDDEN_UNITS_2, 'layer2')

with tf.name_scope('dropout_2'):
    dropped_2 = tf.nn.dropout(hidden2, keep_prob)
    
y_hat = nn_layer(dropped_2, HIDDEN_UNITS_2, OUTPUT_DIM, 'output_layer', tf.identity)

# losses
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels = y, logits = y_hat)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train = tf.train.AdadeltaOptimizer(LEARNING_RATE).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_hat, axis=1), y)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# summary
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'test'))

kf = KFold(n_splits=3, shuffle=True)
total_acc = []

for train_index, val_index in kf.split(X_train):
    
    tf.global_variables_initializer().run()
    
    train_data = X_train[train_index]
    val_data = X_train[val_index]
    train_labels = Y_train[train_index] 
    val_labels = Y_train[val_index]
    
    for i in range(EPOCHS):
        if (i+1) % 10 == 0:
            summary, acc = sess.run([merged, accuracy], 
                                    feed_dict = {
                                        x: X_test.A,
                                        y: Y_test,
                                        keep_prob: 1.0})
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i+1, acc))
        
        summary, _ = sess.run([merged, train], 
                              feed_dict = {
                                    x: X_train.A,
                                    y: Y_train,
                                    keep_prob: KEEP_PROB})
        train_writer.add_summary(summary,i)
        
    train_writer.close()
    test_writer.close()
    # 没有用K折验证
    break
    
k_acc = sess.run(accuracy,
                 feed_dict = {
                    x: val_data.A,
                    y: val_labels,
                    keep_prob: 1.0})

total_acc.append(k_acc)

print('validation accuracy: %.5f' % np.mean(total_acc))

test_acc = sess.run(accuracy,
                     feed_dict = {
                        x: X_test.A,
                        y: Y_test,
                        keep_prob: 1.0})
print('test accuracy: %.5f' % test_acc)