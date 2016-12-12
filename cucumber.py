import tensorflow as tf
from random import shuffle



def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    fdict = cPickle.load(fo)
    fo.close()
    return fdict
    
def main():
    data = unpickle('cucumber_data/p1/data_batch_1')
    data.keys()
    # imgplot = plt.imshow(
    #     np.array(zip(
    #             data['data'][312],
    #             data['data'][313],
    #             data['data'][314]))
    #     .reshape((32, 32, 3)))
    x = tf.placeholder(tf.float32, [None, 3*32*32])
    x_image = tf.reshape(x, [-1,32,32,3])
    y_ = tf.placeholder(tf.float32, [None, 9])
    
    neurons1 = 32
    W_conv1 = weight_variable([2, 2, 3, neurons1])
    b_conv1 = bias_variable([neurons1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    neurons2 = 64
    W_conv2 = weight_variable([2, 2, neurons1, neurons2])
    b_conv2 = bias_variable([neurons2])

    h_conv2 = tf.nn.relu(conv2d(h_p1ool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    neurons3 = 128
    W_conv3 = weight_variable([2, 2, neurons2, neurons3])
    b_conv3 = bias_variable([neurons3])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    neurons4 = 2**9
    W_fc1 = weight_variable([4 * 4 * neurons3, neurons4])
    b_fc1 = bias_variable([neurons4])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*neurons3])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([neurons4, 9])
    b_fc2 = bias_variable([9])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    sess = tf.InteractiveSession()
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    for i in xrange(6000):
        train_cycle(train_step)
        if i % 100 == 0:
            print "Round {}".format(i)
            test_cycle(sess)

def train_cycle(train_step):
    train_order = range(1, 6)
    shuffle(train_order)
    for i in train_order:
        batch_xs, batch_ys = get_batch('cucumber_data/p1/data_batch_{}'.format(i))
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

def test_cycle(sess):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_x, test_y = get_batch('cucumber_data/p1/test_batch')
    print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y, keep_prob: 1.0}))

def flatten_image(image):
    return [item for sublist in image for item in sublist]

def format_data(data):
    images = []
    data_size = len(data)/3
    for i in xrange(data_size):
        images.append(flatten_image(data[3*i:3*i+3]))
    return images

def format_label(labels):
    prep = [np.array([0]*9) for i in xrange(len(labels))]
    for i in xrange(len(labels)):
        prep[i][labels[i]] += 1
    return prep

def get_batch(file):
    data_batch = unpickle(file)
    data = format_data(data_batch['data'])
    labels = format_label(data_batch['labels'])
    return data, labels

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

