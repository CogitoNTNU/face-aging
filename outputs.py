import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def write_log(callback, name, value, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()
    
def save_rgb_img(img, path):
    """
    Save an rgb image
    """
    fig = plt.figure(frameon=False)
    #ax = fig.add_subplot(1, 1, 1)
    fig.set_size_inches(1,1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.imshow(((img + 1) * 127.5).astype(np.int16))
    ax.set_axis_off()
    fig.add_axes(ax)
    #ax.axis("off")
    #ax.set_title("Image")
    
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi = img.shape[0])
    plt.close()