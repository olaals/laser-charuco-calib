import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from utils import *
import rhovee as rv

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




def stereo_matching(left_img, right_img, max_disp=64, block_size=5):
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max_disp, blockSize=block_size)
    disparity = stereo.compute(left_img, right_img)
    return disparity

def cross_prod_mat(mat):
    mat = mat.squeeze()
    return np.array([[0, -mat[2], mat[1]], [mat[2], 0, -mat[0]], [-mat[1], mat[0], 0]])


def get_essential_matrix(R_12, t_12):
    T = np.eye(4)
    T[:3,:3] = R_12
    T[:3,3] = t_12
    T_inv = np.linalg.inv(T)
    R_21 = T_inv[:3,:3]
    t_21 = T_inv[:3,3]
    E = cross_prod_mat(t_21)@(R_21)
    return E

def homg_points_to_plucker_line(pt1, pt2):
    pt1 = pt1.squeeze()
    pt2 = pt2.squeeze()
    if pt1.shape[0] == 3:
        pt1 = np.append(pt1, 1)
    if pt2.shape[0] == 3:
        pt2 = np.append(pt2, 1)
    l = pt1[3]*pt2[:3] - pt2[3]*pt1[:3]
    l_dash = np.cross(pt1[:3],pt2[:3])
    return l, l_dash

def intersect_plucker_lines(l1, l1_dash, l2, l2_dash):
    n = np.cross(l1, l2)
    v = np.zeros(4)
    v[:3] = np.cross(n, l1)
    v[3] = np.dot(n, l1_dash)
    x = np.zeros(4)
    x[:3] = -v[3]*l2+np.cross(v[:3], l2_dash)
    x[3] = np.dot(v[:3], l2)
    x = x / x[3]
    return x[:3]





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
    t = t/1000.0
    print("R", R)
    print("t", t)
    print("Rotation axis angle degrees")
    print(rotation_mat_to_axis_angle(R) * 180 / np.pi)

    # load laser images
    left_laser_paths = glob.glob('laser-images/test-images/left/*.png')
    right_laser_paths = glob.glob('laser-images/test-images/right/*.png')
    left_laser_paths.sort()
    right_laser_paths.sort()
    all_pts = []

    for left_laser_path,right_laser_path in zip(left_laser_paths, right_laser_paths):
        limg = cv2.imread(left_laser_path, 0)
        rimg = cv2.imread(right_laser_path, 0)
        lK = l_cam_mat
        ldc = l_dist_coeffs
        rK = r_cam_mat
        rdc = r_dist_coeffs
        points = rv.cv.triangulate_laser_lines(limg, rimg, 50, R, t, lK, ldc, rK, rdc, 0)
        all_pts.append(points)
        """
        left_laser_img = cv2.imread(left_laser_path, 0)
        # undistort the images
        left_laser_img = cv2.undistort(left_laser_img, l_cam_mat, l_dist_coeffs)
        #left_laser_img = process_laser_img(left_laser_img)
        right_laser_img = cv2.imread(right_laser_path, 0)
        right_laser_img = cv2.undistort(right_laser_img, r_cam_mat, r_dist_coeffs)
        l_homg = rv.cv.get_laser_line_as_homg(left_laser_img, 50, 0)
        r_homg = rv.cv.get_laser_line_as_homg(right_laser_img, 50, 0)
        l_mean_rows = rv.cv.weighted_mean_row_index(left_laser_img, 50, 0)
        l_points = rv.cv.row_list_to_points(l_mean_rows)
        print("l_points", l_points)
        E = get_essential_matrix(R, t)
        F = np.linalg.inv(r_cam_mat.T) @ E @ np.linalg.inv(l_cam_mat)
        zero_img = right_laser_img.copy()

        for idx,point in enumerate(l_points):
            if idx%20 == 0:
                #l_point = np.array([point[0], point[1], 1])
                line = F @ point
                # intersect r_homg and line
                intersect_r = np.cross(r_homg, line)
                intersect_r = intersect_r / intersect_r[2]
                norm_r = np.linalg.inv(r_cam_mat) @ intersect_r
                norm_r = norm_r / norm_r[2]
                # transform to left camera frame
                norm_l = R @ norm_r + t
                print("t", t)
                print("norm_l shape", norm_l.shape)
                print("t shape", t.shape)
                line_r, line_r_dash = homg_points_to_plucker_line(norm_l, t)
                pl_line_l = np.linalg.inv(l_cam_mat) @ point
                line_l = pl_line_l
                line_l_dash = np.zeros(3)
                # intersect line_r and line_l
                x = intersect_plucker_lines(line_r, line_r_dash, line_l, line_l_dash)
                print("x", x)
                all_pts.append(x)
                
        """


                




                #img_line = rv.cv.draw_homg_line(zero_img, line)
                #cv2.imshow("img_line", img_line)
                #kcv2.waitKey(0)
    # visualize the points
    all_pts = np.vstack(all_pts)
    print("all_pts", all_pts)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2])
    # label xyz axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()








    







    

