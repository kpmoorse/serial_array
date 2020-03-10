import numpy as np
import cv2
import warnings

# Initialize device-agnostic pipe
pipe = "v4l2src device={} do-timestamp=true "
pipe += "! image/jpeg, width=1920, height=1080, framerate=30/1 "
pipe += "! jpegparse ! jpegdec ! videoconvert ! appsink"

# Generate device-specific caps
dev_list = ['/dev/video0', '/dev/video1']
cap_list = [cv2.VideoCapture(pipe.format(dev), cv2.CAP_GSTREAMER) for dev in dev_list]

while(True):

    # Capture frame-by-frame
    ret_sum = True
    frame_list = []
    ret_list = []
    for cap in cap_list:
        ret, frame = cap.read()
        ret_list.append(ret)
        ret_sum = ret_sum & ret
        frame_list.append(frame)

    # Check returns and skip frame if capture failed
    if not all(ret_list):
        msg = "Frame capture failed on "
        for dev, ret in zip(dev_list, ret_list):
            if not ret:
                msg += dev+' '
        warnings.warn(msg)
        continue

    disp = np.hstack([cv2.pyrDown(cv2.pyrDown(frame,2)) for frame in frame_list])

    # Display the resulting frame
    cv2.imshow('frame',disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
for cap in cap_list:
    cap.release()
cv2.destroyAllWindows()