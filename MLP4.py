from __future__ import division
import os
import re
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score,f1_score,recall_score,accuracy_score
import math

clean_pat = re.compile(r'\w{2,}')
pos_path = os.getcwd() + "\pos"
pos_test_path = os.getcwd() + "\\test_pos"
neg_test_path = os.getcwd() + "\\test_neg"
neg_path = os.getcwd() + "\\neg"
pos_dict = {}
neg_dict = {}
count_pos = 0
count_neg = 0

def getWordCount(dir):
    temp_count = {}
    count = 0
    for filename in os.listdir(dir):
        with open(dir+ "\\" +filename,'r') as files:
            count += 1
            for i in files.readlines():
                for word in re.findall(clean_pat,i):
                    if not word in temp_count:
                        temp_count[word] = 1
                    else:
                        temp_count[word]+= 1
    return temp_count

def convertInput(str):
    temp = []
    if str == "pos":
        temp = [1, 0]
    if str == "neg":
        temp = [0, 1]
    return temp

def generate_vocab_list(input_pos_dict,input_neg_dict):
    vocab_list = []
    for word in input_pos_dict:
        temp = (input_pos_dict[word],"pos")
        vocab_list.append(temp)
    for word in input_neg_dict:
        temp = (input_neg_dict[word],"neg")
        vocab_list.append(temp)
    return vocab_list

def convertVocab(vocab):
    input_count = []
    classes = []
    for i in vocab:
        input_count.append([i[0]])
        temp = convertInput(i[1])
        classes.append(temp)
    return input_count,classes

def asArray(input,classes):
    input_lab = np.asarray(input, dtype=np.float)
    classes = np.asarray(classes, dtype=np.float)
    return input_lab,classes

def MLP():
    input_train_count, input_train_class = convertVocab(vocab_train_list)
    input_test_count, input_test_class = convertVocab(vocab_test_list)

    # Parameters
    learning_rate = 0.001
    training_epochs = 2000
    batch_size = 100
    display_step = 200

    # Network Parameters
    n_hidden_1 = 20
    n_hidden_2 = 20
    n_hidden_3 = 20

    n_input = 1
    n_classes = 2

    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Create model
    def multilayer_perceptron(x):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)

        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3 = tf.nn.sigmoid(layer_3)
        out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
        return out_layer

    # Construct model
    logits = multilayer_perceptron(X)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.

            _, c = sess.run([train_op, loss_op], feed_dict={X: input_train_count,
                                                                Y: input_train_class})
                # Compute average loss
            avg_cost += c / len(input_train_count)
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        pred = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({X: input_test_count, Y: input_test_class}))
        y_p = tf.argmax(pred, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={X: input_test_count, Y: input_test_class})

        print ("validation accuracy:", val_accuracy)
        y_true = np.argmax(input_test_class, 1)
        print("Precision", precision_score(y_true, y_pred))
        print("Recall", recall_score(y_true, y_pred))
        print("f1_score", f1_score(y_true, y_pred))



pos_dict = getWordCount(pos_path)
neg_dict= getWordCount(neg_path)
pos_test_dict = getWordCount(pos_test_path)
neg_test_dict = getWordCount(neg_test_path)
vocab_train_list = generate_vocab_list(pos_dict,neg_dict)
vocab_test_list = generate_vocab_list(pos_test_dict,neg_test_dict)
MLP()



