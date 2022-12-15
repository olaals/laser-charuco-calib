import numpy as np
import cv2
import os


def weighted_mean_row_index(img):
    out = []
    for row in range(img.shape[0]):
        # find index weighted mean
        indeces = np.arange(img.shape[1])
        mean = np.sum(indeces * img[row, :]) / np.sum(img[row, :])
        out.append(mean)
    return np.array(out)

def weighted_mean_row_index_vectorized(img, threshold=20):
    img = np.where(img > threshold, img, 0)
    # find index weighted mean
    indeces = np.arange(img.shape[1])
    mean = np.sum(indeces * img, axis=1) / np.sum(img, axis=1)
    return mean

def row_list_to_points(row_list):
    ys = np.arange(len(row_list))
    xs = row_list
    return np.vstack([xs, ys, np.ones(len(xs))]).T


def pix_coords_to_points(coords):
    coords = np.swapaxes(coords, 0, 1)
    coords = np.vstack([coords, np.ones(coords.shape[1])])
    return coords


def row_indices_to_img(img, row_indices):
    out = np.zeros_like(img)
    for row in range(img.shape[0]):
        out[row, int(row_indices[row])] = 255
    return out

def determine_best_fit_line(points):
    # find m,c using least squares
    # points are of (n,3) and are on the form [x,y,1]
    # m = (x' * y - x * y') / (x' * x - x * x')
    # c = (x' * x * y' - x * x' * y) / (x' * x - x * x')
    x = points[0, :]
    y = points[1, :]
    x_ = np.mean(x)
    y_ = np.mean(y)
    m = (np.sum(x * y) - np.sum(x) * np.sum(y)) / (np.sum(x * x) - np.sum(x) * np.sum(x))
    c = (np.sum(x * x) * np.sum(y) - np.sum(x) * np.sum(x) * np.sum(y)) / (np.sum(x * x) - np.sum(x) * np.sum(x))
    return m, c


def m_c_to_homogeneous(m, c):
    return np.array([m, -1, c])

def draw_line(img, m,c):
    out = img.copy()
    x1 = 0
    y1 = int(m * x1 + c)
    x2 = img.shape[1]
    y2 = int(m * x2 + c)
    cv2.line(out, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return out

def draw_homg_line(img, l):
    out = img.copy()
    x1 = 0
    y1 = int(-l[2] / l[1])
    x2 = img.shape[1]
    y2 = int(-(l[2] + l[0] * x2) / l[1])
    cv2.line(out, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return out

def fit_homogenous_line(points):
    assert points.shape[1] == 3
    # find m,c using least squares
    points = np.asarray(points)
    n, _ = points.shape
    mean_x = np.mean(points[:,0])
    mean_y = np.mean(points[:,1])
    sum_xy = sum([points[i,0]*points[i,1] for i in range(n)])
    sum_x = sum([points[i,0] for i in range(n)])
    sum_y = sum([points[i,1] for i in range(n)])
    sum_x2 = sum([points[i,0]**2 for i in range(n)])
    slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)
    intercept = mean_y - slope*mean_x
    return [-slope, 1, -intercept]



if __name__ == '__main__':
    img_path = "laser-images/test-images/left/img00.png"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.where(img>50, img, 0)

    #show img
    #cv2.imshow("img", img)
    #cv2.waitKey(0)
    w_coords = weighted_mean_row_index_vectorized(img)
    w_points = row_list_to_points(w_coords)
    print(w_points.shape)
    # find best fit line
    zero_img = np.zeros_like(img)
    l = fit_homogenous_line(w_points)
    print(l)
    zero_img = draw_homg_line(zero_img, l)
    # overlap zero_img and img in rgb
    out = np.zeros((img.shape[0], img.shape[1], 3))
    out[:, :, 0] = img
    out[:, :, 1] = zero_img


    cv2.imshow("img", out)
    cv2.waitKey(0)



