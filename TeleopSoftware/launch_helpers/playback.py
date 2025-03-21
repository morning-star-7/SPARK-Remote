import rclpy
from rclpy.node import Node
import rosbag2_py
import argparse
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Int32, String
from rclpy.serialization import deserialize_message
from datetime import datetime, timedelta
import numpy as np

class BagPlayback(Node):
    def __init__(self, bag_filename):
        super().__init__('bag_playback')
        self.bag_filename = bag_filename
        self.reader = rosbag2_py.SequentialReader()
        self.bridge = CvBridge()
        self.last_ft_messages_thunder = []
        self.last_ft_messages_lightning = []
        self.thunder_enable = False
        self.start_time = None
        self.elapsed_time = timedelta(0)
        self.max_torque_thunder = 0
        self.max_torque_lightning = 0 
        self.safety_mode_thunder = 1
        self.safety_mode_lightning = 1

    def start_playback(self):
        try:
            storage_options = rosbag2_py.StorageOptions(uri=self.bag_filename, storage_id='sqlite3')
            converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
            self.reader.open(storage_options, converter_options)
        except Exception as e:
            self.get_logger().error(f"Failed to open bag file: {e}")
            return

        topic_types = self.reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        while self.reader.has_next():
            (topic, data, t) = self.reader.read_next()
            msg_type = self.get_message_type(type_map[topic])
            if msg_type is None:
                continue
            msg = deserialize_message(data, msg_type)

            if topic == '/video_frames':
                self.display_video_frame(msg)
            elif topic == '/lightning_enable':
                self.update_thunder_enable(msg)
            elif topic.endswith('thunder_ft'):
                self.save_ft_message(msg, 'thunder')
            elif topic.endswith('lightning_ft'):
                self.save_ft_message(msg, 'lightning')
            elif topic.endswith('safety_mode'):
                self.update_safety_mode(msg, topic)

    def get_message_type(self, type_str):
        if type_str == 'sensor_msgs/msg/Image':
            return Image
        elif type_str == 'std_msgs/msg/Bool':
            return Bool
        elif type_str == 'std_msgs/msg/Float32MultiArray':
            return Float32MultiArray
        elif type_str == 'std_msgs/msg/Int32':
            return Int32
        elif type_str == 'std_msgs/msg/String':
            return String
        return None

    def display_video_frame(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Make the frame brighter
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=0)

        if self.thunder_enable:
            if self.start_time is None:
                self.start_time = datetime.now()
            self.elapsed_time += datetime.now() - self.start_time
            self.start_time = datetime.now()
        else:
            self.start_time = None

        # Devide the frame into the left and right. 
        # If the saftey mode is not 1, make the left frame red with alpha 0.1
        left_frame = frame[:, :frame.shape[1] // 2]
        right_frame = frame[:, frame.shape[1] // 2:]

        if self.safety_mode_thunder != 1:
            red_overlay = np.full_like(left_frame, (0, 0, 255))
            left_frame = cv2.addWeighted(left_frame, 1.0, red_overlay, 0.2, 0)

        if self.safety_mode_lightning != 1:
            red_overlay = np.full_like(right_frame, (0, 0, 255))
            right_frame = cv2.addWeighted(right_frame, 1.0, red_overlay, 0.2, 0)

        frame = np.hstack((left_frame, right_frame))


        timestamp = f"{self.elapsed_time.total_seconds():.2f}"
        font_scale = 4  # Increased font scale
        font_thickness = 8  # Increased font thickness
        text_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = 130  # Initial text position
        offset = 550  # Offset to move text lower
        cv2.rectangle(frame, (text_x - 20, text_y - 100 + offset), (text_x + text_size[0] + 20, text_y + 20 + offset), (0, 0, 0), -1)  # Adjusted rectangle size
        cv2.putText(frame, timestamp, (text_x, text_y + offset),  # Adjusted text position
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
            (255, 255, 255), font_thickness, 
            cv2.LINE_AA,
            False)

        # Draw vertical bars for torque values
        max_force = 170.0
        bar_height_thunder = int((self.max_torque_thunder / max_force) * (frame.shape[0] - 20))  # Allow a gap at the bottom and top
        bar_height_lightning = int((self.max_torque_lightning / max_force) * (frame.shape[0] - 20))  # Allow a gap at the bottom and top
        bar_width = 50
        gap = 10  # Gap at the bottom and top

        # Determine color based on torque value
        color_thunder = (0, 255, 0) if self.max_torque_thunder < max_force * 0.5 else (0, 255, 255) if self.max_torque_thunder < max_force * 0.75 else (0, 0, 255)
        color_lightning = (0, 255, 0) if self.max_torque_lightning < max_force * 0.5 else (0, 255, 255) if self.max_torque_lightning < max_force * 0.75 else (0, 0, 255)

        cv2.rectangle(frame, (10, frame.shape[0] - bar_height_thunder - gap), (10 + bar_width, frame.shape[0] - gap), color_thunder, -1)
        cv2.rectangle(frame, (frame.shape[1] - 10 - bar_width, frame.shape[0] - bar_height_lightning - gap), (frame.shape[1] - 10, frame.shape[0] - gap), color_lightning, -1)

        cv2.imshow('Video Frame', frame)
        cv2.waitKey(1)

    def update_thunder_enable(self, msg):
        # print(msg.data)
        self.thunder_enable = msg.data

    def save_ft_message(self, msg, arm):
        if arm == 'thunder':
            self.last_ft_messages_thunder.append(msg)
            if len(self.last_ft_messages_thunder) > 10:  # Keep only the last 10 messages
                self.last_ft_messages_thunder.pop(0)
            self.max_torque_thunder = max(abs(torque) for torque in msg.data)
        elif arm == 'lightning':
            self.last_ft_messages_lightning.append(msg)
            if len(self.last_ft_messages_lightning) > 10:  # Keep only the last 10 messages
                self.last_ft_messages_lightning.pop(0)
            self.max_torque_lightning = max(abs(torque) for torque in msg.data)

    def update_safety_mode(self, msg, topic):
        arm = 'thunder' if 'thunder' in topic else 'lightning'
        if arm == 'thunder':
            self.safety_mode_thunder = msg.data
        else:
            self.safety_mode_lightning = msg.data
        # self.get_logger().info(f"Safety mode for {arm}: {msg.data}")

def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description="ROS 2 Bag Playback")
    parser.add_argument("filename", type=str, help="Filename for the ROS 2 bag")
    args = parser.parse_args()

    node = BagPlayback(args.filename)

    try:
        # input("Press Enter to start playback...")
        node.start_playback()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()