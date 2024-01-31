import tensorflow_datasets as tfds
import tensorflow as tf

import numpy as np

class NoiseScheduler:

    def prepare_mnist_data(self, mnist):
        
        mnist = mnist.filter(lambda img, label: label == 0) # only '0' digits

        mnist = mnist.map(lambda img, label: img )

        mnist = mnist.map(lambda img: tf.cast(img, tf.float32) )
        mnist = mnist.map(lambda img: (img/128.)-1. )
        
        mnist = mnist.map(lambda img: tf.image.resize(img, [32,32]) )

        mnist = mnist.batch(64, drop_remainder=True)
        mnist = mnist.prefetch(tf.data.experimental.AUTOTUNE)
 
        return mnist
    
    def __init__(self):
        self.ds = tfds.load('mnist', split="train+test", as_supervised=True)

        self.ds = self.ds.apply(self.prepare_mnist_data)

        self.noise_steps = 250
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.beta = tf.linspace(self.beta_start, self.beta_end, self.noise_steps)

        # (250, 1024)
        self.encoding_time_steps = self.positional_encoding(length=self.noise_steps, depth=32*32)

        # (250, 32, 32, 1)
        self.encoding_time_steps = tf.reshape(self.encoding_time_steps, shape=(-1, 32, 32, 1))
    
        # (250, 1, 32, 32, 1)
        self.encoding_time_steps = tf.expand_dims(self.encoding_time_steps, axis=1)

        # (250, 64, 32, 32, 1)
        self.encoding_time_steps = tf.tile(self.encoding_time_steps, multiples=[1,64,1,1,1])

    
    def positional_encoding(self, length, depth):
        depth = depth/2

        positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

        angle_rates = 1 / (10000**depths)         # (1, depth)
        angle_rads = positions * angle_rates      # (pos, depth)

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1) 

        return tf.cast(pos_encoding, dtype=tf.float32)   
    

    
    def sample(self, u_net, x):
        
        imgs = []
        for t, beta in enumerate(self.beta[::-1]):
            t = self.noise_steps - t - 1

            t_encoded = self.encoding_time_steps[t][:1, ...]

            x_t_embedded = tf.concat([x, t_encoded], axis=-1)
            noise = u_net(x_t_embedded)
            
            x = 1/(tf.math.sqrt(1 - beta)) * (x - tf.math.sqrt(beta)*noise)
            
            imgs.append(x)
        
        return imgs
        

    def dataset_generator(self):
         for x in self.ds:
            
            batch_size = x.shape[0]
            for t, beta in enumerate(self.beta):

                noise = tf.random.normal(shape=x.shape) 
                x = tf.math.sqrt(1 - beta) * x + tf.math.sqrt(beta) * noise

                t_encoded = self.encoding_time_steps[t]
                noised_x_t_embedded = tf.concat([x, t_encoded], axis=-1)

                for i in range(batch_size):
                    yield noised_x_t_embedded[i], noise[i]




    def prepare_data(self, ds):
    
        ds = ds.cache()

        ds = ds.shuffle(5000)
        ds = ds.batch(64)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
 
        return ds
    
    def create_dataset(self):

        ds = tf.data.Dataset.from_generator(
                self.dataset_generator,
                args=(),
                output_signature=(
                    tf.TensorSpec(shape=(32,32,2), dtype=tf.float32),
                    tf.TensorSpec(shape=(32,32,1), dtype=tf.float32),
                )
            )
        
        ds = ds.apply(self.prepare_data)
        
        return ds