import random
from itertools import product
import torch

def createMask(img_rgb, points = 250, hor_num_squares=4, ver_num_squares=4):
  c, h, w = img_rgb.shape

  total_squares = hor_num_squares * ver_num_squares
  
  hor_size_square= w // hor_num_squares  # size of the square in the horizontal axis
  ver_size_square= h // ver_num_squares  # size of the square in the vertical axis

  points_per_square = points // total_squares # num puntos por cada cuadrado (16 en total)
  
  # Initializing the mask
  mask = torch.zeros([c, h, w])

  for i in range(0, hor_num_squares):
      for j in range(0,ver_num_squares):
          x_y = [[_x, _y] for (_x, _y) in product(range(i*hor_size_square,hor_size_square*(i+1)), 
                                                  range(j*ver_size_square,ver_size_square*(j+1)))]
          rand_x_y = random.sample(x_y,points_per_square)

          for x, y in rand_x_y:
            # Extract color
            color = img_rgb[:, y, x]
            # Assign color to the mask
            mask[:, y, x] = color

  return mask


def imageIn0to1Torch(img):
  min_img = torch.min(img)
  max_img = torch.max(img)

  rng = max_img - min_img

  return (img - min_img)/rng