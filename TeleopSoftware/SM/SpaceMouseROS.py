import pyspacemouse
import time
import os
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray

class SpaceMouseROS(Node):
    def __init__(self, dev):
        super().__init__('SpaceMouse' + dev[-1])
        self.dev = dev
        self.thunder = self.create_publisher(Float32MultiArray, 'SpaceMouseThunder', 1)
        self.thunder_log = self.create_publisher(String, 'SpaceMouseThunderLog', 10)
        self.lightning = self.create_publisher(Float32MultiArray, 'SpaceMouseLightning', 1)
        self.lightning_log = self.create_publisher(String, 'SpaceMouseLightningLog', 10)
        self.success = pyspacemouse.open(path=self.dev)
        self.mod = 50
        self.count = 0
        self.pub = None
        self.log = None

    def run(self):
        if self.success:
            while rclpy.ok():
                time.sleep(0.0001)
                state = pyspacemouse.read()
                self.count += 1
                if self.pub is None:
                    if state.buttons[0] == 1:
                        self.pub = self.thunder
                        self.log = self.thunder_log
                        self.log.publish(String(data="Thunder connected to SpaceMouse: " + self.dev))
                        self.get_logger().info("Thunder connected to SpaceMouse: " + self.dev)
                    elif state.buttons[1] == 1:
                        self.pub = self.lightning
                        self.log = self.lightning_log
                        self.log.publish(String(data="Lightning connected to SpaceMouse: " + self.dev))
                        self.get_logger().info("Lightning connected to SpaceMouse: " + self.dev)
                    self.count = 0
                else:
                    if self.count == self.mod:
                        self.count = 0
                        data = [state.x, state.y, state.z, state.roll, state.pitch, state.yaw] + state.buttons
                        self.pub.publish(Float32MultiArray(data=data))

def main(args=None):
    rclpy.init(args=args)
    dev = os.sys.argv[1]
    print("Connecting to SpaceMouse: " + dev)
    sys.stdout = open(os.devnull, "w")
    node = SpaceMouseROS(dev)
    sys.stdout = sys.__stdout__
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()