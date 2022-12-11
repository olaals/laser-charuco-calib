import cv2
import numpy as np
from utils import filter_hsv, get_images_from_dir, read_json, write_json


def filter_hsv_gui(img, lower=[0,0,0], upper=[179,255,255]):
    hmin = lower[0]
    hmax = upper[0]
    smin = lower[1]
    smax = upper[1]
    vmin = lower[2]
    vmax = upper[2]
    def nothing(x):
        pass
    cv2.namedWindow('image')
    cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
    cv2.createTrackbar('SMin','image',0,255,nothing)
    cv2.createTrackbar('VMin','image',0,255,nothing)
    cv2.createTrackbar('HMax','image',0,179,nothing)
    cv2.createTrackbar('SMax','image',0,255,nothing)
    cv2.createTrackbar('VMax','image',0,255,nothing)
    cv2.setTrackbarPos('HMax', 'image', hmax)
    cv2.setTrackbarPos('SMax', 'image', smax)
    cv2.setTrackbarPos('VMax', 'image', vmax)
    cv2.setTrackbarPos('HMin', 'image', hmin)
    cv2.setTrackbarPos('SMin', 'image', smin)
    cv2.setTrackbarPos('VMin', 'image', vmin)
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0
    output = img
    waitTime = 33
    while(1):
        hMin = cv2.getTrackbarPos('HMin','image')
        sMin = cv2.getTrackbarPos('SMin','image')
        vMin = cv2.getTrackbarPos('VMin','image')
        hMax = cv2.getTrackbarPos('HMax','image')
        sMax = cv2.getTrackbarPos('SMax','image')
        vMax = cv2.getTrackbarPos('VMax','image')
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
        output = filter_hsv(img, lower,upper)
        if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax
        cv2.imshow('image',output)
        if cv2.waitKey(waitTime) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return [hMin, sMin, vMin], [hMax, sMax, vMax]

def filter_imgs_in_dir(image_dir, lower, upper):
    image_paths = get_images_from_dir(image_dir)
    for image_path in image_paths:
        img = cv2.imread(image_path)
        assert len(img.shape) == 3, "Image must be RGB"
        lower, upper = filter_hsv_gui(img, lower, upper)
    return lower, upper




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Filter HSV')
    # add image dir argument
    parser.add_argument('--image_dir', type=str, help='image directory', required=True)
    # --lower with default value [0,0,0]
    parser.add_argument('--lower', type=int, nargs=3, default=[0,0,0], help='lower bound of HSV')
    # --upper with default value [179,255,255]
    parser.add_argument('--upper', type=int, nargs=3, default=[179,255,255], help='upper bound of HSV')
    # write to json
    parser.add_argument('--write_json', type=str, help='write to json', required=True)





    args = parser.parse_args()
    lower = args.lower
    upper = args.upper
    lower = [int(i) for i in lower]
    upper = [int(i) for i in upper]
    lower, upper = filter_imgs_in_dir(args.image_dir, lower, upper)
    # from numpy array to list
    print("Lower_hsv", lower)
    print("Upper_hsv", upper)
    if args.write_json:
        json_dict = read_json(args.write_json)
        json_dict['lower_hsv'] = np.array(lower).tolist()
        json_dict['upper_hsv'] = np.array(upper).tolist()
        write_json(args.write_json, json_dict)






