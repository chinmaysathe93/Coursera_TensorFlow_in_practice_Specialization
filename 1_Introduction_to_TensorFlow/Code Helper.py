!pip install tensorflow==2.0.0-alpha0 

activate nlp_course
pip install --upgrade tensorflow

# ------------------------------------------
Rule of Thumbs :

#1 By adding more Neurons we have to do more calculations, slowing down the process,that doesn't 
#mean it's always a case of 'more is better'.

#2 The first layer in your network should be the same shape as your data.

#3 The number of neurons in the last layer should match the number of classes you are classifying for. 

#4 Normalize the Data before inputing

# ------------------------------------------
# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>0.998):
                print("\n Reached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True
    
    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    # YOUR CODE STARTS HERE
    training_images=training_images.reshape(60000, 28, 28, 1)
    test_images=test_images.reshape(10000, 28, 28, 1)
    training_images, test_images = training_images / 255.0, test_images / 255.0
    callbacks = myCallback()
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
            # YOUR CODE STARTS HERE
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            # YOUR CODE ENDS HERE
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(
        # YOUR CODE STARTS HERE
    training_images, training_labels, epochs=20, callbacks=[callbacks]
        # YOUR CODE ENDS HERE
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]
# ------------------------------------------

