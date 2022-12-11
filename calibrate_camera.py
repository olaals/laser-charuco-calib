import cv2
from cv2 import aruco
import numpy as np
import glob
import os
from utils import load_board_from_dict, get_images_from_dir, read_json, write_json


def calibrate_camera(image_paths, board, req_markers=10):
    all_corners = []
    all_ids = []
    for im_path in images:
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








if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Calibrate camera')
    parser.add_argument('--img_dir', type=str, help='Directory with images', required=True)
    parser.add_argument('--board_dir', type=str, help='Board directory', required=True)
    parser.add_argument('--show_imgs', type=bool, default=False, help='Show images')
    # json_name
    parser.add_argument('--json_name', type=str, default='latest_calib', help='Json name')







    args = parser.parse_args()
    board_json = os.path.join(args.board_dir, 'board_dict.json')
    board_dict= read_json(board_json)
    board = load_board_from_dict(board_dict)
    args = parser.parse_args()
    images = get_images_from_dir(args.img_dir)
    cam_mat, dist_coeffs = calibrate_camera(images, board)

    calib_dict = {
        'cam_mat': cam_mat.tolist(),
        'dist_coeffs': dist_coeffs.tolist()
    }
    calib_json = os.path.join("calib-results", args.json_name+".json")
    write_json(calib_json, calib_dict)




