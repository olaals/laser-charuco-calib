import numpy as np
import cv2
from cv2 import aruco
from utils import write_json, read_json, create_board, filter_hsv, get_images_from_dir, load_board_from_dict
import os
import open3d as o3d


def fit_plane_svd(points):
    assert points.shape[0] >= 3
    assert points.shape[1] == 3
    mean = np.mean(points, axis=0)
    U, S, Vt = np.linalg.svd(points - mean)
    normal = Vt[-1]
    d = -normal.dot(mean)
    return np.concatenate((normal, [d]))

def intersect_points(norm_img_coords, plane):
    n,d = plane
    t = -d / np.dot(n, norm_img_coords)
    points = norm_img_coords * t
    return points

def transf_z_to_plane(T):
    assert T.shape == (4,4)
    n = T[:3,2]
    d = -np.dot(T[:3,3], n)
    return n, d

def image_to_normalized_img_coords(mask, K):
    assert len(mask.shape) == 2
    assert np.all(np.logical_or(mask, np.logical_not(mask))), "mask must be binary boolean"
    h,w = mask.shape
    x = np.arange(w)
    y = np.arange(h)
    xv, yv = np.meshgrid(x,y)
    xv = xv[mask]
    yv = yv[mask]
    uv = np.vstack((xv,yv,np.ones_like(xv)))
    uv = np.linalg.inv(K) @ uv
    return uv

def mask_image(image, min_intensity):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = image > min_intensity
    return mask

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

def calibrate_laser(image_dir, K, dist_coeffs, board, lower_hsv, upper_hsv):
    # get images
    img_paths = get_images_from_dir(image_dir)
    all_points = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        T = get_charuco_cb_pose(img, board, K, dist_coeffs)
        if T is not None:
            plane = transf_z_to_plane(T)
            filtered_img = filter_hsv(img, lower_hsv, upper_hsv)
            filtered_img = cv2.undistort(filtered_img, K, dist_coeffs)
            filtered_img = np.mean(filtered_img, axis=2)
            mask = np.where(filtered_img>0, True, False)
            norm_img_coords = image_to_normalized_img_coords(mask, K)
            intersected_points = intersect_points(norm_img_coords, plane)
            print("Intersected points: ", intersected_points.shape)
            all_points.append(intersected_points)
    # stack all_points
    all_points = np.hstack(all_points).T
    u = fit_plane_svd(all_points)
    print("all_points.shape", all_points.shape)
    return u, all_points


def visualize_points_and_plane_o3d(points_np):
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(points_np)
    o3d.visualization.draw_geometries([points])








if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Calibrate the laser')
    parser.add_argument('--board_json', type=str, help='Board description', required=True)
    parser.add_argument('--calib_json', type=str, help='Calibration json file', required=True)
    parser.add_argument('--image_dir', type=str, help='Directory with images', required=True)

    args = parser.parse_args()
    calib = read_json(args.calib_json)
    K = np.array(calib['cam_mat'])
    dist_coeffs = np.array(calib['dist_coeffs'])
    lower_hsv = np.array(calib['lower_hsv'])
    upper_hsv = np.array(calib['upper_hsv'])
    # print loaded Calibration
    print("K", K)
    print("dist_coeffs", dist_coeffs)
    print("lower_hsv", lower_hsv)
    print("upper_hsv", upper_hsv)
    # load board json
    board_dict = read_json(args.board_json)
    print("board_dict", board_dict)
    board = load_board_from_dict(board_dict)
    u, all_points = calibrate_laser(args.image_dir, K, dist_coeffs, board, lower_hsv, upper_hsv)
    u[-1] = u[-1]/1000.0
    print("u", u)
    # read json

    calib["u"] = u.tolist()

    write_json(args.calib_json, calib)








