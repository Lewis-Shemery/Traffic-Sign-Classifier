
################################
#### L O A D  D A T A S E T ####
################################

import pickle
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np

# TODO: fill in quotes with file path of traffic sign data
training_file = 'data/train.p'
validation_file= 'data/validate.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = np.array(train['features']), np.array(train['labels'])
X_valid, y_valid = np.array(valid['features']), np.array(valid['labels'])
X_test, y_test = np.array(test['features']), np.array(test['labels'])

# Let's get some dimensions
print("Features shape: ", X_train.shape)
print("Labels shape: ", y_train.shape)

# Number of training examples
n_train = X_train.shape[0]

# Number of validation examples
n_validation = X_valid.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# Show one of each sign class
img_is = []
label_num = 0
for i, label in enumerate(y_train):
    if label == label_num:
        img_is.append(i)
        label_num += 1

signs = X_train[img_is]

for img in signs:
    plt.figure(i + 1)
    plt.imshow(img)
    plt.show()
    
# Display histograms of class frequencies
plt.hist(y_train, bins=n_classes)
plt.title("Train Labels")
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.show()

plt.hist(y_test, bins=n_classes)
plt.title("Test Labels")
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.show()

# TEST print out images matched with labels
label_names = [
    'Speed limit (20km/h)',
    'Speed limit (30km/h)',
    'Speed limit (50km/h)',
    'Speed limit (60km/h)',
    'Speed limit (70km/h)',
    'Speed limit (80km/h)',
    'End of speed limit (80km/h)',
    'Speed limit (100km/h)',
    'Speed limit (120km/h)',
    'No passing',
    'No passing for vechiles over 3.5 metric tons',
    'Right-of-way at the next intersection',
    'Priority road',
    'Yield',
    'Stop',
    'No vechiles',
    'Vechiles over 3.5 metric tons prohibited',
    'No entry',
    'General caution',
    'Dangerous curve to the left',
    'Dangerous curve to the right',
    'Double curve',
    'Bumpy road',
    'Slippery road',
    'Road narrows on the right',
    'Road work',
    'Traffic signals',
    'Pedestrians',
    'Children crossing',
    'Bicycles crossing',
    'Beware of ice/snow',
    'Wild animals crossing',
    'End of all speed and passing limits',
    'Turn right ahead',
    'Turn left ahead',
    'Ahead only',
    'Go straight or right',
    'Go straight or left',
    'Keep right',
    'Keep left',
    'Roundabout mandatory',
    'End of no passing',
    'End of no passing by vechiles over 3.5 metric tons'
]

def show_imgs_labels(imgs, labels, num=5):
    indeces = np.random.choice(len(imgs), num, replace=False)
    for i in indeces:
        label = label_names[labels[i]]
        
        img = imgs[i]
        if img.shape[2] == 1:
            img = img[:, :, 0]
        
        print(label)
        plt.figure(i)
        plt.imshow(img)
        plt.show()


##############################################
#### P R E - P R O C E S S  D A T A S E T ####
##############################################

# convert to greyscale
def rgb2gray(imgs):
    print('runing rgb2gray...')
    return np.mean(imgs, axis=3, keepdims=True)

# normalize pixel values inclusively between -1 and 1
def normalize(imgs):
    print('runing normalize...')
    return (imgs-128) / 128

# equalize contrast
# https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
# Greedy algorithm for local contrast enhancement of images
def equalize(imgs):
    print('runing equalize...')
    new_imgs = np.empty(imgs.shape, dtype=float)
    for i, img in enumerate(imgs):
        equalized_img = exposure.equalize_adapthist(img) * 2 - 1
        new_imgs[i] = equalized_img

    return new_imgs
    
def preprocess(imgs):
    print('runing_preprocess...')
    new_imgs = equalize(imgs)
    new_imgs = rgb2gray(new_imgs)
    
    return new_imgs

# preprocess the images
X_train_processed = preprocess(X_train)
X_valid_processed = preprocess(X_valid) # what's the point of this?
X_test_processed = preprocess(X_test)

# generate a numpy array of randomly selected images and their corresponding labels
# zip() generates an iterator of tuples based on the values passed
inputs_train_valid, labels_train_valid = map(np.array, zip(*np.random.permutation(list(zip(X_train_processed, y_train)))))

# visualize new images
show_imgs_labels(inputs_train_valid, labels_train_valid)

# finding a number equal to 10% of the number of training images
split_i = int(len(inputs_train_valid) * 0.1)

inputs_validation = inputs_train_valid[:split_i]
labels_validation = labels_train_valid[:split_i]

inputs_train = inputs_train_valid[split_i:]
labels_train = labels_train_valid[split_i:]

# for naming consistency
inputs_test = X_test_processed
labels_test = y_test 

show_imgs_labels(inputs_train, labels_train)

# help with understanding this
def weight(shape, seed=None):
    print('runing_weight...')
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=tf.float32)

# help with understanding this
def bias(shape, const=0.1):
    print('runing_bias...')
    return tf.Variable(tf.constant(const, shape=shape))


############################################
#### M O D E L  A R C H I T E C T U R E ####
############################################
'''
CNN with 3 layers that have 32, 64, and 128 feature maps respectively. 
'''

import tensorflow as tf

class Model:
    def __init__(self, sess, lrate):
        self.sess = sess
        self.lrate = lrate
        
        self.define_graph()
        
    def define_graph(self):
        print('runing define_graph...')
        with tf.name_scope('Data'):
            self.inputs = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], 1])
            self.labels = tf.placeholder(tf.uint8)
            self.labels_onehot = tf.one_hot(self.labels, n_classes)
        
        with tf.name_scope('Variables'):
            self.ws_conv = []
            self.bs_conv = []
            self.ws_fc   = []
            self.bs_fc   = []

            conv_fms = [1, 32, 64, 128]
            filter_sizes = [5, 5, 3] #is this filter size okay?

            fc_sizes = [2048, 512, n_classes]
            
            with tf.name_scope('Conv'):
                for layer in range(len(filter_sizes)):
                    with tf.name_scope(str(layer)):
                        self.ws_conv.append(weight([filter_sizes[layer],
                                               filter_sizes[layer],
                                               conv_fms[layer],
                                               conv_fms[layer + 1]]))
                        self.bs_conv.append(bias([conv_fms[layer + 1]]))
                        
            with tf.name_scope('FC'):
                for layer in range(len(fc_sizes) - 1):
                    with tf.name_scope(str(layer)):
                        self.ws_fc.append(weight([fc_sizes[layer], fc_sizes[layer + 1]]))
                        self.bs_fc.append(bias([fc_sizes[layer + 1]]))
                        
        with tf.name_scope('Training'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.get_logits(self.inputs), labels=self.labels_onehot))
            
            ################################
            ## A D A M  O P T I M I Z E R ##
            ################################
            self.global_step = tf.Variable(0, trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lrate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            
            self.preds = self.get_preds(self.inputs)
            self.accuracy, self.update_accuracy_op = tf.contrib.metrics.streaming_accuracy(
                tf.argmax(self.preds, 1), self.labels)
            
    def get_logits(self, inputs):
        print('runing get_logits...')
        with tf.name_scope('Calculation'):
            logits = inputs
            
            with tf.name_scope('Conv'):
                for layer, (kernel, bias) in enumerate(zip(self.ws_conv, self.bs_conv)):
                    with tf.name_scope(str(layer)):
                        logits = tf.nn.conv2d(logits, kernel, [1, 1, 1, 1], 'SAME') + bias
                        logits = tf.nn.max_pool(logits, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
                        logits = tf.nn.relu(logits)
                        
            # flatten logits
            shape = tf.shape(logits)
            logits = tf.reshape(logits, [shape[0], shape[1] * shape[2] * shape[3]])
            
            with tf.name_scope('FC'):
                for layer, (weight, bias) in enumerate(zip(self.ws_fc, self.bs_fc)):
                    with tf.name_scope(str(layer)):
                        logits = tf.matmul(logits, weight) + bias
                        
                        # Activate with ReLU if not the last layer
                        if layer < len(self.ws_fc) - 1:
                            logits = tf.nn.relu(logits)
        
        return logits
    
    def get_preds(self, inputs):
        print('runingget_preds...')
        return tf.nn.softmax(self.get_logits(inputs))
    
    def train(self, inputs, labels):
        feed_dict = {self.inputs: inputs, self.labels: labels}
        loss, global_step, _ = self.sess.run(
            [self.loss, self.global_step, self.train_op], feed_dict=feed_dict)
        
        if global_step % 10 == 0:
            print('Step {} | Loss: {}'.format(global_step, loss))
            
        if global_step % 1000 == 0:
            self.test(inputs_validation, labels_validation)
            
        return global_step
    
    def test(self, inputs, labels):
        print('runing test...')
        print('-' * 30)
        batch_gen = gen_epoch(inputs, labels, BATCH_SIZE)
        
        total_loss = 0
        for step, (inputs, labels) in enumerate(batch_gen):
            feed_dict = {self.inputs: inputs, self.labels: labels}
            loss, preds, _ = self.sess.run([self.loss, self.preds, self.update_accuracy_op], feed_dict=feed_dict)
            
            total_loss += loss
            
            if step % 10 == 0:
                print('TEST | Step {} | Loss: {}'.format(step, loss))
            
        avg_loss = total_loss / float(step)
        accuracy = self.sess.run([self.accuracy])
        print('FINAL | LOSS: {} | ACCURACY: {}'.format(avg_loss, accuracy))
        print('-' * 30)
        
        return avg_loss, accuracy 
    
# Helpers
def gen_epoch(inputs, labels, batch_size):
    print('runing gen_epoch...')
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        
        yield batch_inputs, batch_labels

# Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 10
LRATE = 0.001

# Initialization
sess = tf.Session()
model = Model(sess, LRATE)
sess.run([tf.initialize_all_variables(), tf.initialize_local_variables()])

saver = tf.train.Saver()

# train
print('training...')
for epoch in range(NUM_EPOCHS):
    batch_gen = gen_epoch(inputs_train, labels_train, BATCH_SIZE)
    for (inputs, labels) in batch_gen:
        step = model.train(inputs, labels)
        
        # save the model
        if step % 1000 == 0:
            save_path = saver.save(sess, "models/ ") # TODO: enter some save path once dataset is downloaded
            print("Model saved in file: %s" % save_path)


model.test(inputs_validation, labels_validation)