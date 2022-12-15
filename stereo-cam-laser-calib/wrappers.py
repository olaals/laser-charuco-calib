from harvesters.core import Harvester
import cv2
import numpy as np
import os


class GenicamRGB():
    def __init__(self, cti_path, device_num=0, exposure_time=60000, width=2176, height=2176):
        h = Harvester()
        h.add_file(cti_path)
        h.update()
        h.device_info_list[0]
        print(h.device_info_list)
        ia = h.create(device_num)
        # set exposure time
        #ia.remote_device.node_map.ExposureTime.value = exposure_time
        # set width
        
        ia.remote_device.node_map.Width.value = width
        # get max width
        max_width = ia.remote_device.node_map.Width.max
        max_width = 4112
        print("max_width", max_width)
        # calcualte center x
        offset_x = (max_width-width)//2
        print("offset_x", offset_x)
        # calculate offset x such that image is centered
        # center x
        ia.remote_device.node_map.OffsetX.value = offset_x

        ia.start()
        self.ia=ia


    def read(self):
        with self.ia.fetch() as buffer:
            component = buffer.payload.components[0]
            _1d = component.data
            _2d = component.data.reshape(component.height, component.width)
            img_2d = np.array(_2d)
            debayered = cv2.cvtColor(img_2d, cv2.COLOR_BAYER_BG2BGR)
            resized = cv2.resize(debayered, (component.width//2, component.height//2))
            print("resized shape", resized.shape)
            return resized

class GenicamStereoRGB():
    def __init__(self, cti_path):
        h = Harvester()
        h.add_file(cti_path)
        h.update()
        h.device_info_list[0]
        ia1 = h.create(0)
        ia1.start()
        ia2= h.create(1)
        ia2.start()
        self.ia1=ia1
        self.ia2=ia2
    
    def read(self):
        with self.ia1.fetch() as buffer:
            component = buffer.payload.components[0]
            _1d = component.data
            _2d = component.data.reshape(component.height, component.width)
            img_2d = np.array(_2d)
            debayered = cv2.cvtColor(img_2d, cv2.COLOR_BAYER_BG2BGR)
            resized1 = cv2.resize(debayered, (component.width//2, component.height//2))
        with self.ia2.fetch() as buffer:
            component = buffer.payload.components[0]
            _1d = component.data
            _2d = component.data.reshape(component.height, component.width)
            img_2d = np.array(_2d)
            debayered = cv2.cvtColor(img_2d, cv2.COLOR_BAYER_BG2BGR)
            resized2 = cv2.resize(debayered, (component.width//2, component.height//2))
        return (resized1, resized2)


class GenicamPolar():
    def __init__(self):
        h = Harvester()
        h.add_file('/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti')
        h.update()
        h.device_info_list[0]
        ia = h.create(0)
        ia.start()
        self.ia=ia

    def read(self):
        with self.ia.fetch() as buffer:
            component = buffer.payload.components[0]
            _1d = component.data
            _2d = component.data.reshape(component.height, component.width)
            img_2d = np.array(_2d)
            red_0 = img_2d[1::4,1::4]
            red_45 = img_2d[0::4,1::4]
            red_90 = img_2d[0::4,0::4]
            red_135 = img_2d[1::4,0::4]
            return (red_0, red_45, red_90, red_135)

class GenicamTest():
    def __init__(self):
        h = Harvester()
        h.add_file('/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti')
        h.update()
        h.device_info_list[0]
        ia = h.create(0)
        ia.start()
        self.ia=ia

    def read(self):
        with self.ia.fetch() as buffer:
            component = buffer.payload.components[0]
            _1d = component.data
            _2d = component.data.reshape(component.height, component.width)
            img_2d = np.array(_2d)
            r = img_2d[0::4,0::4]
            g = img_2d[3::4,1::4]
            b = img_2d[0::4,3::4]
            return np.dstack([b,g,r])

def capture_stereo():
    cti_path = '/opt/cvb-13.04.001/drivers/genicam/libGevTL.cti'
    cap = GenicamStereoRGB(cti_path)
    left_save_path = 'laser-images/test-images/left'
    right_save_path = 'laser-images/test-images/right'
    save_index = 0
    while(True):
        (img1,img2) = cap.read()
        cv2.imshow("main", img1)
        cv2.imshow("main2", img2)

        c = cv2.waitKey(1)
        if c == 27:
            break
        # save if s is pressed
        if c == ord('s'):
            cv2.imwrite(os.path.join(left_save_path, "img"+format(save_index, '02d')+'.png'), img1)
            cv2.imwrite(os.path.join(right_save_path, "img"+format(save_index, '02d')+'.png'), img2)
            save_index += 1




            


if __name__ == '__main__':
    capture_stereo()



