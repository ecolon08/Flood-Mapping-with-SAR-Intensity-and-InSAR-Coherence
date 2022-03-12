import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.colors

def gen_cmap(clr_lst):
    # convert hex to rgb
    colors = [matplotlib.colors.to_rgb(i) for i in clr_lst]

    # create cmap object
    return ListedColormap(colors)


def display_image_target(display_list, cmap):
  plt.figure(dpi=200)
  title = ['Image', 'Target', 'Prediction']

  for idx, disp in enumerate(display_list):
    plt.subplot(1, len(display_list), idx+1)
    plt.title(title[idx], fontsize=6)
    plt.axis('off')

    if title[idx] == 'Image':
      arr = disp.numpy()

      min = np.min(arr, axis=(0, 1))
      max = np.max(arr, axis=(0, 1))
      arr = (arr - min) / (max - min)

      if arr.shape[-1] > 3:

        plt.imshow(arr[:, :, 0], cmap='gray')
      else:
          #plt.imshow(arr, cmap='gray')
          plt.imshow(arr[:, :, 0], cmap='gray')

    elif title[idx] == 'Target':
      tgt = disp.numpy().squeeze()
      plt.imshow(tgt, interpolation='none', cmap=cmap)

    elif title[idx] == 'Prediction':
      pred = np.argmax(disp, axis=-1) # argmax across probabilities to get class outputs
      plt.imshow(pred, interpolation='none', cmap=cmap)

  plt.show()
  plt.close()
