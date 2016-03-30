import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def fused_lasso_2d(Y, lambda1):
    T = np.zeros(Y.shape)
    for i in xrange(Y.shape[0], 0, -1):
        for j in xrange(Y.shape[1], 0, -1):
            if j == Y.shape[1]:
                right = Y[i-1, j-1]
            else:
                right = T[i-1, j]

            if i == Y.shape[0]:
                down = Y[i-1, j-1]
            else:
                down = T[i, j-1]

            p, q = min(right, down), max(right, down)
            y = Y[i-1,j-1]
            if y < p - 2*lambda1:
                T[i-1,j-1] = y + 2*lambda1
            elif y <= p:
                T[i-1,j-1] = p
            elif y < q:
                T[i-1,j-1] = y
            elif y <= q + 2*lambda1:
                T[i-1,j-1] = q
            else: # q + 2*lambda1 < y
                T[i-1,j-1] = y - 2*lambda1
    return T

def preprocess(img):
    a = np.array(img, dtype=myfloat) / 256
    mean = a.mean()
    return (mean, a-mean)

def convert_to_image(mean, T):
    a =  np.clip((T + mean) * 256, 0, 255)
    return Image.fromarray(a)

myfloat = np.float64
lambda1 = 0.01
src_img = Image.open('fused_lasso_orig_lena.png')
mean, Y = preprocess(src_img)
T = fused_lasso_2d(Y, lambda1)
dst_img = convert_to_image(mean, T)
dst_img.convert('L').save('fused_lasso_denoised_lena.png')
