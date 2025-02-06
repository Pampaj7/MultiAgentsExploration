import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def create_video(n):
    global X
    X = np.random.binomial(1, 0.3, size = (n, n))

    fig = plt.figure()
    im = plt.imshow(X, cmap = plt.cm.gray)

    def animate(t):
        global X
        X = np.roll(X, +1, axis = 0)
        im.set_array(X)
        return im,

    anim = FuncAnimation(
        fig,
        animate,
        frames = 100,
        interval = 1000/30,
        blit = True
    )

    plt.show()

    return anim

anim = create_video(100000)