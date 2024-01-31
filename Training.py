
import tensorflow as tf

import matplotlib.pyplot as plt

import datetime
import tqdm

from NoiseScheduler import *

from UNet import *


def main():

    
    #
    # Logging
    #
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"logs/{current_time}"
    summary_writer = tf.summary.create_file_writer(file_path)

    noiseScheduler = NoiseScheduler()


    ds = noiseScheduler.create_dataset()

    u_net = UNet()
    u_net.build(input_shape=(1, 32, 32, 2))
    u_net.summary()

    for epoch in range(10000):
        
        print(f"Epoch: {epoch}")
        for x, noise_target in tqdm.tqdm(ds):
            u_net.train_step(x, noise_target)

        loss = u_net.metric_loss.result()
        print(f"Loss: {loss}")
        with summary_writer.as_default():
            tf.summary.scalar(name="Loss", data=loss, step=epoch)

        u_net.metric_loss.reset_states()

        if epoch % 5 == 0:
            u_net.save_weights(f"./saved_models/trained_weights_{epoch}", save_format="tf")
    

    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")