import rclpy
from rclpy.node import Node
import rosbag2_py
import argparse
import sys

class BagRecorder(Node):
    def __init__(self, bag_filename):
        super().__init__('bag_recorder')
        self.bag_filename = bag_filename
        self.recorder = rosbag2_py.Recorder()
        self.recording = False

    def start_recording(self):
        if not self.recording:
            self.get_logger().info(f"Starting recording to {self.bag_filename}")
            storage_options = rosbag2_py.StorageOptions(uri=self.bag_filename, storage_id='sqlite3')
            record_options = rosbag2_py.RecordOptions()
            record_options.all_topics = True  # Record all topics
            self.recorder.record(storage_options=storage_options, record_options=record_options)
            self.recording = True

    def stop_recording(self):
        if self.recording:
            self.get_logger().info("Stopping recording")
            self.recorder.stop()
            self.recording = False

def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description="ROS 2 Bag Recorder")
    parser.add_argument("filename", type=str, help="Filename for the ROS 2 bag")
    args = parser.parse_args()

    node = BagRecorder(args.filename)

    try:
        input("Press Enter to start recording...")
        node.start_recording()
        print("Recording... Press Ctrl+C to stop.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_recording()
    finally:
        node.destroy_node()
        # rclpy.shutdown()

if __name__ == "__main__":
    main()