import tensorflow as tf

class UNet(tf.keras.Model):
    def __init__(self):

        super(UNet, self).__init__()

        self.encoder_layers = [
            tf.keras.layers.Conv2D(8, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(40, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu'),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2 * 2 * 40, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu')
        ]

        self.decoder_layers = [
            tf.keras.layers.Dense(2 * 2 * 40, activation='relu'),
            tf.keras.layers.Reshape((2, 2, 40)), 

            tf.keras.layers.Conv2DTranspose(40, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(16, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(8, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
            
            tf.keras.layers.Conv2DTranspose(1, kernel_size=(3,3), strides=(1,1), padding='same', activation='tanh'),
        ]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss_function = tf.keras.losses.MeanSquaredError()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")

    @tf.function
    def call(self, x):

        encoder_activations = []
        for layer in self.encoder_layers:
            x = layer(x)

            if not isinstance(layer, tf.keras.layers.Flatten):
                encoder_activations.append(x)

        encoder_activations = encoder_activations[:-1]
        encoder_activations = encoder_activations[::-1]
        
    
        layer_cnt = 0
        for layer in self.decoder_layers:
            
            if not isinstance(layer, tf.keras.layers.Reshape) and layer_cnt < len(encoder_activations):
                x = tf.concat([x, encoder_activations[layer_cnt]], axis=-1)
                layer_cnt += 1

            x = layer(x)

        return x
    
    @tf.function
    def train_step(self, x, noise_target):
  
        with tf.GradientTape() as tape:
            noise_pred = self(x)
            loss = self.loss_function(noise_target, noise_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_loss.update_state(loss)

        return loss
    