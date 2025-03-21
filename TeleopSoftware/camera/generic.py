import rospy
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
import sys
import subprocess
# import compressed_image_transport
import theora_image_transport as compressed_image_transport

def get_camera_info():
    cameras = []
    for i in range(10):
        device = f"/dev/video{i}"
        try:
            output = subprocess.check_output(["v4l2-ctl", "--device", device, "--all"], stderr=subprocess.STDOUT).decode()
            if "Cannot open device" not in output:
                cameras.append((i, output))
        except subprocess.CalledProcessError:
            continue
    return cameras

def print_camera_info(cameras):
    for index, info in cameras:
        print(f"Camera Index: {index}")
        print(info)
        print("="*40)

try:
    if len(sys.argv) < 2:
        camera_uid = None
    else:
        camera_uid = sys.argv[1]
    rospy.init_node("USB_Camera", anonymous=True)

    cameras = get_camera_info()
    if not cameras:
        raise Exception("No cameras found")

    # print_camera_info(cameras)

    camera_index = None
    if camera_uid is not None:
        for index, info in cameras:
            if camera_uid in info:
                camera_index = index
                break
        if camera_index is None:
            raise Exception(f"Camera with UID {camera_uid} not found")
    else:
        camera_index = cameras[0][0]

    hz = 10
    rate = rospy.Rate(hz)
    # RGB_topic = "/cameras/rgb/compressed"
    RGB_topic = f"/cameras/rgb/theora"
    pub_rgb = rospy.Publisher(RGB_topic, CompressedImage, queue_size=10)
    bridge = CvBridge()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise Exception(f"Cannot open camera {camera_index}")

    working = False
    num_frames = 1

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        msg = bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')
        pub_rgb.publish(msg)

        if not working:
            working = True
            print(f"\tPublishing RGB frames from camera: {camera_uid}")

        rate.sleep()
        if num_frames % (hz * 20) == 0:
            print(f"\tFrames published({camera_uid}): \t{num_frames}")
        num_frames += 1

    cap.release()

except Exception as e:
    print(e)
    pass