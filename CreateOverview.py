import tensorflow as tf

from NoiseScheduler import *

from UNet import *
import matplotlib.pyplot as plt

def main():
    u_net = UNet()
    u_net.build(input_shape=(1, 32, 32, 2))
    u_net.load_weights("./saved_models/trained_weights_80").expect_partial()
    u_net.summary()

    noiseScheduler = NoiseScheduler()

    fig, axs = plt.subplots(figsize=(15, 7), nrows=5, ncols=26)
   
    x = tf.random.normal(shape=(5, 32,32,1))

    imgs = noiseScheduler.sample(u_net, x)

    for t in range(26):
        time_step = t*10
        if time_step == 250:
            time_step = 249
        
        img = imgs[time_step]

        for batch_idx in range(5):
            axs[batch_idx, t].imshow(img[batch_idx])

            axs[batch_idx, t].axis("off")

            if batch_idx == 0:
                axs[batch_idx, t].set_title(time_step)
    
    plt.tight_layout()
    plt.savefig("Overview.png")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")