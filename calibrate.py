import numpy as np
import cv2
from time import time
import csv


class Calibrate(object):

    def __init__(self, dev='/dev/video0', res=(1920,1080)):

        self.calib = {}
        self.res = res
        self.dev = dev
        self.generate_pipe()
        self.cap = cv2.VideoCapture(self.pipe, cv2.CAP_GSTREAMER)

    # Run calibration loop (circle grid)
    def calibrate(self, n=15):

        # Generate object points
        x = -np.tile(np.array([0,2,4,6,8,-1,1,3,5,7])[:,None],[4,1])
        y = -np.tile(np.arange(0,8)[:,None],[1,5]).reshape([-1,1])
        objp = np.hstack((y,x,np.zeros(x.shape)))[:35,:].astype(np.float32)

        objpoints = []
        imgpoints = []
        fl = 0 # Flash value
        ts = time() # Timestamp

        while len(objpoints) < n:
            ret, frame = self.cap.read()
            if not ret:
                print("Frame capture failed")
                continue

            # Preprocess frame for maximal contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            window = self.res[0]//50*2+1
            gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,window,-20)

            # Find circle centers and draw to frame
            dim = (5,7)
            ret, corners = cv2.findCirclesGrid(gray, dim, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            
            if ret == True:
                if fl == 0:
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    fl = 1
                    print("Iterations: {}/{}".format(len(objpoints), n))
                cv2.drawChessboardCorners(frame, dim, corners, ret)

            frame = self.flash(frame,fl)
            disp = np.vstack((frame, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
            fl = max(fl - (time()-ts)*0.5, 0)
            ts = time()

            cv2.imshow('Frame',cv2.pyrDown(disp))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Calculate calibration coefficients
        if len(objpoints)>0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            print("ret = "+str(ret))
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, self.res, 1, self.res)
        self.calib = {
            "c_old": mtx,
            "c_new": newcameramtx,
            "dist": dist,
            "roi": roi
        }

    @staticmethod
    def flash(img, I):
        
        img = 255*I + (1-I)*img
        return img.astype('uint8')

    def store_params(self, fname='params.txt'):

        with open(fname, 'w') as file:
            writer = csv.writer(file, delimiter=',')

            writer.writerow(self.calib["c_old"].flatten())
            writer.writerow(self.calib["c_new"].flatten())
            writer.writerow(self.calib["dist"].flatten())
            writer.writerow(self.calib["roi"])

    def retrieve_params(self, fname='params.txt'):

        with open(fname, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            headers = ["c_old", "c_new", "dist", "roi"]
            for i, row in enumerate(reader):
                if i in [0,1]: # Camera matrices
                    self.calib[headers[i]] = np.array(row, dtype=float).reshape([3,3])
                elif i == 2: # Distortion coefficients
                    self.calib[headers[i]] = np.array(row, dtype=float)
                elif i == 3: # ROI pixel locations
                    self.calib[headers[i]] = np.array(row, dtype=int)
        
    def test_calibration(self):

        # if not self.calib:
        #     raise ValueError("Calibration not defined")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Frame capture failed")
                continue

            cal = cv2.undistort(
                frame,
                self.calib["c_old"],
                self.calib["dist"],
                None,
                self.calib["c_new"]
            )
            print(self.calib["roi"])
            x, y, w, h = self.calib["roi"]
            # cal = cal[y:y+h, x:x+w]
            cal_disp = np.zeros(cal.shape, dtype='uint8')
            cal_disp[y:y+h, x:x+w] = cal[y:y+h, x:x+w]
            # print(cal)

            disp = cv2.pyrDown(np.hstack((frame, cal)))

            cv2.imshow('Frame',disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Generate pipe from device name
    def generate_pipe(self):

        pipe = "v4l2src device={} do-timestamp=true "
        pipe += "! image/jpeg, width={}, height={}, framerate=30/1 "
        pipe += "! jpegparse ! jpegdec ! videoconvert ! appsink"
        
        self.pipe = pipe.format(self.dev, self.res[0], self.res[1])


if __name__ == '__main__':

    cal = Calibrate('/dev/video1')
    # cal = Calibrate('/dev/video1', res=(1280,720))
    cal.calibrate()
    cal.store_params()
    cal.test_calibration()