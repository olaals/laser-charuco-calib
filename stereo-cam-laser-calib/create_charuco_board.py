import os
import cv2
from cv2 import aruco
import numpy as np
from utils import create_board

A4_Y = 195.0
A4_X = 276.0
A4_XY_RATIO = A4_X/A4_Y

def create_printable_aruco_grid(aruco_dict_str, px_width, squares_x,squares_y, spacing_ratio, start_id, padding):
    #aruco_dict.bytesList=aruco_dict.bytesList[start_id:,:,:]
    squares_xy_ratio = squares_x/squares_y
    px_per_mm = px_width/A4_X
    print("px per mm", px_per_mm)
    #board = aruco.CharucoBoard_create(squares_x,squares_y,1,spacing_ratio,aruco_dict)
    board = create_board(squares_x, squares_y, 1, spacing_ratio, aruco_dict_str, start_id)
    px_height = np.round(px_width/A4_XY_RATIO, 0)
    if squares_xy_ratio > A4_XY_RATIO:
        norm_width = (squares_x*spacing_ratio+squares_x)
        #padding = px_width*1.0/norm_width*spacing_ratio/2
        img = board.draw((px_width,int(px_height)), marginSize=int(padding))
        ch_board_sq_size = ((px_width-2*padding)/squares_x)/px_per_mm
        aruko_size = spacing_ratio*ch_board_sq_size
        print("ch_board_size", ch_board_sq_size)
        print("aruko_size", aruko_size)
    else:
        norm_height = ((squares_y+1)*spacing_ratio+squares_y)
        #padding = px_height*1.0/norm_height*spacing_ratio
        img = board.draw((px_width,int(px_height)), marginSize=int(padding))
        ch_board_sq_size = ((px_height-2*padding)/squares_y)/px_per_mm
        aruko_size = spacing_ratio*ch_board_sq_size
        print("ch_board_size", ch_board_sq_size)
        print("aruko_size", aruko_size)
    font_scale = px_width/1000.0
    font_thickness = px_width//500
    label = "APRILTAG_16H5" + f' SZ_CH_SQ:{np.round(ch_board_sq_size, 3)}mm'
    label += f' AR_SZ:{np.round(aruko_size, 3)}mm' + f' start id: {str(start_id)}'
    imboard = cv2.putText(img, label, (100,100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=font_scale, color=(0,0,0), thickness=font_thickness)
    img = img.T
    img = cv2.flip(img,1)
    aruco_board_dict={
        "aruco_dict_str":aruco_dict_str,
        "square_x":squares_x,
        "square_y":squares_y,
        "cb_sq_width":ch_board_sq_size,
        "aruco_sq_width":aruko_size,
        "start_id":start_id,
    }
    return img, aruco_board_dict

if __name__ == '__main__':
    import argparse
    import json
    # create option for aruco_dict_str
    parser = argparse.ArgumentParser()
    parser.add_argument('--aruco_dict_str', type=str, default='DICT_APRILTAG_16H5')
    parser.add_argument('--sq_x', type=int, default=7)
    parser.add_argument('--sq_y', type=int, default=6)
    parser.add_argument('--spacing_ratio', type=float, default=0.65)
    parser.add_argument('--padding', type=int, default=800)
    parser.add_argument('--px_width', type=int, default=4000)
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--min_brightness', type=int, default=128)
    args = parser.parse_args()

    img, aruco_board_dict = create_printable_aruco_grid(args.aruco_dict_str, args.px_width, args.sq_x, args.sq_y, args.spacing_ratio, args.start_id, args.padding)
    board_save_dir = f'{args.aruco_dict_str}_{args.sq_x}x{args.sq_y}_start_id_{args.start_id}_{args.padding}'
    img = np.where(img < args.min_brightness, args.min_brightness, img)
    board_save_dir = os.path.join("boards", board_save_dir)
    if not os.path.exists(board_save_dir):
        os.makedirs(board_save_dir)
    cv2.imwrite(os.path.join(board_save_dir, "board.png"), img)
    # save as json
    with open(os.path.join(board_save_dir, "board_dict.json"), "w") as f:
        # with nice print
        json.dump(aruco_board_dict, f, indent=4)
    print("Saving to", board_save_dir)








    


