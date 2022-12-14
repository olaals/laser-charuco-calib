import os 
import numpy as np
import glob
import cv2
import cv2.aruco as aruco
import json

def get_images_from_dir(dir_path):
    """Get all images from a directory"""
    images = []
    for ext in ['jpg', 'jpeg', 'png']:
        images.extend(glob.glob(os.path.join(dir_path, '*.{}'.format(ext))))
    return images

def create_board(squares_x, squares_y, cb_sq_width, aruco_sq_width, aruco_dict_str, start_id):
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_str))
    aruco_dict.bytesList=aruco_dict.bytesList[start_id:,:,:]
    board = aruco.CharucoBoard_create(squares_x,squares_y,cb_sq_width,aruco_sq_width,aruco_dict)
    return board

def load_board_from_dict(board_dict):
    squares_x = board_dict['square_x']
    squares_y = board_dict['square_y']
    cb_sq_width = board_dict['cb_sq_width']
    aruco_sq_width = board_dict['aruco_sq_width']
    aruco_dict_str = board_dict['aruco_dict_str']
    start_id = board_dict['start_id']
    return create_board(squares_x, squares_y, cb_sq_width, aruco_sq_width, aruco_dict_str, start_id)


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)

def get_K_distcoeffs_from_json(json_path):
    data = read_json(json_path)
    K = np.array(data['cam_mat']).astype(np.float32)
    distcoeffs = np.array(data['dist_coeffs']).astype(np.float32)
    return K, distcoeffs

def filter_hsv(img, lower=[0,0,0], upper=[179,255,255]):
    lower = np.array(lower)
    upper = np.array(upper)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img,img, mask= mask)
    return output


