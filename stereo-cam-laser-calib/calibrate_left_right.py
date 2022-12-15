import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from utils import *

def calibrate_camera(image_paths, board, req_markers=10):
    all_corners = []
    all_ids = []
    for im_path in image_paths:
        im_col = cv2.imread(im_path)
        if len(im_col.shape) == 3:
            gray = cv2.cvtColor(im_col, cv2.COLOR_BGR2GRAY)
        else:
            gray = im_col
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, board.dictionary)
        if len(corners) > 0 and len(corners) >= req_markers:
            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if ret is not None:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
    print("Found {} images with at least {} markers".format(len(all_corners), req_markers))
    if len(all_corners) > 0:
        print("Calibrating camera...")
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(all_corners, all_ids, board, gray.shape, None, None)
        print("Camera calibration complete.")
        return camera_matrix, dist_coeffs
    else:
        print("No images with enough markers found.")
        return None, None, None, None, None

def rvec_to_vec_to_transf(rvec,tvec):
    R = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3,:3] = R[0]
    T[:3,3] = tvec.squeeze()
    return T

def transf_to_rvec_to_vec(T):
    rvec = cv2.Rodrigues(T[:3,:3])[0]
    tvec = T[:3,3]
    return rvec, tvec


def get_charuco_cb_pose(img, board, K, dist_coeffs, req_det_markers=6):
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, board.dictionary)
    if ids is not None and len(ids) >= req_det_markers:
        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, img, board)
        if charuco_corners is not None:
            retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, K, dist_coeffs, np.array([]), np.array([]))
            if retval:
                # convert to transformation matrix
                T = rvec_to_vec_to_transf(rvec,tvec)
                return T
    return None


def draw_frame_axes(img, K, dist_coeffs, T, size=10):
    img = img.copy()
    rvec, tvec = transf_to_rvec_to_vec(T)
    cv2.drawFrameAxes(img, K, dist_coeffs, rvec, tvec, size)
    return img

def average_transformations(Ts):
    avg_T = np.eye(4)
    for T in Ts:
        avg_T = avg_T.dot(T)
    avg_T = avg_T / len(Ts)
    return avg_T

def average_rotation_matrices_svd(Rs):
    avg_R = np.zeros((3,3))
    for R in Rs:
        avg_R = avg_R + R
    avg_R = avg_R / len(Rs)
    U, S, V = np.linalg.svd(avg_R)
    avg_R = U.dot(V)
    return avg_R

def average_translation_vectors(ts):
    avg_t = np.zeros(3)
    for t in ts:
        avg_t = avg_t + t
    avg_t = avg_t / len(ts)
    return avg_t

def average_transformation_matrices(Ts):
    Rs = []
    ts = []
    for T in Ts:
        Rs.append(T[:3,:3])
        ts.append(T[:3,3])
    avg_R = average_rotation_matrices_svd(Rs)
    avg_t = average_translation_vectors(ts)
    avg_T = np.eye(4)
    avg_T[:3,:3] = avg_R
    avg_T[:3,3] = avg_t
    return avg_T


def rotation_mat_to_axis_angle(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    w = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
    w = w / np.linalg.norm(w)
    w = w * theta
    return w

def process_laser_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 0)
    # keep values over 50
    img = np.where(img > 50, img, 0)
    return img

def row_wise_mean(img):
    return np.mean(img, axis=1)


def stereo_matching(left_img, right_img, max_disp=64, block_size=5):
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max_disp, blockSize=block_size)
    disparity = stereo.compute(left_img, right_img)
    return disparity





if __name__ == '__main__':
    board_json = "boards/DICT_APRILTAG_16H5_7x6_start_id_0_1000/board_dict.json"
    board_dict = read_json(board_json)
    board = load_board_from_dict(board_dict)

    left_image_paths = glob.glob('calib-images/test-images/left/*.png')
    right_image_paths = glob.glob('calib-images/test-images/right/*.png')
    left_image_paths.sort()
    right_image_paths.sort()
    l_cam_mat, l_dist_coeffs = calibrate_camera(left_image_paths, board)
    print(l_cam_mat)
    r_cam_mat, r_dist_coeffs = calibrate_camera(right_image_paths, board)
    print(r_cam_mat)
    l_to_r_Ts = []
    for l_path, r_path in zip(left_image_paths, right_image_paths):
        l_img = cv2.imread(l_path)
        r_img = cv2.imread(r_path)
        l_T = get_charuco_cb_pose(l_img, board, l_cam_mat, l_dist_coeffs)
        r_T = get_charuco_cb_pose(r_img, board, r_cam_mat, r_dist_coeffs)
        # calculate the relative pose of the left camera to the right camera
        l_to_r_T = l_T@np.linalg.inv(r_T)
        l_to_r_Ts.append(l_to_r_T)
        print("")
        print(l_to_r_T)
    avg_l_to_r_T = average_transformation_matrices(l_to_r_Ts)
    print("Average left to right transformation matrix:")
    print(avg_l_to_r_T)
    R = avg_l_to_r_T[:3,:3]
    t = avg_l_to_r_T[:3,3]
    print("Rotation axis angle degrees")
    print(rotation_mat_to_axis_angle(R) * 180 / np.pi)

    # load laser images
    left_laser_paths = glob.glob('laser-images/test-images/left/*.png')
    right_laser_paths = glob.glob('laser-images/test-images/right/*.png')
    left_laser_paths.sort()
    right_laser_paths.sort()

    for left_laser_path,right_laser_path in zip(left_laser_paths, right_laser_paths):
        left_laser_img = cv2.imread(left_laser_path)
        left_laser_img = process_laser_img(left_laser_img)
        right_laser_img = cv2.imread(right_laser_path)
        right_laser_img = process_laser_img(right_laser_img)

        #left_laser_img = cv2.cvtColor(left_laser_img, cv2.COLOR_GRAY2BGR)
        #left_laser_img = draw_frame_axes(left_laser_img, l_cam_mat, l_dist_coeffs, avg_l_to_r_T)
        disparity = stereo_matching(left_laser_img, right_laser_img)
        cv2.imshow('left laser', left_laser_img)
        cv2.waitKey(0)
        # show disparity    
        cv2.imshow('disparity', disparity)

    







    

