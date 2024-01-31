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

    x = tf.random.normal(shape=(1, 32,32,1))

    imgs = noiseScheduler.sample(u_net, x)

    for t, img in enumerate(imgs):
        plt.imshow(img[0])
        plt.savefig(f"./results/{t}.png")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")