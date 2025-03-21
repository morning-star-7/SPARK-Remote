import os
import subprocess
import time
import atexit
# import rospy
import rclpy
from rclpy.node import Node


# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9029686
class LaunchDevs(Node):
    def __init__(self):
        super().__init__('LaunchDevs')
        self.get_logger().info("Starting LaunchDevs")

    def get_devs(self):
        cmd = "udevadm info --name="
        devs = []
        spark_devs = []
        SM_devs = []
        VR_devs = []
        haptic_devs = []
        for dev in os.listdir('/dev'):
            if 'ttyACM' in dev:
                devs.append(dev)
        for dev in devs:
            cmd = "udevadm info --name=/dev/" + dev + " --attribute-walk"
            output = os.popen(cmd).read()
            if "STMicroelectronics" in output:
                print("Haptic Device found: " + dev)
                haptic_devs.append(os.path.join("/dev/", dev))
        return spark_devs, SM_devs, VR_devs, haptic_devs

    def cleanup(self, modules, arms):
        for module in modules:
            module.kill()
        print("Exiting")

    def StartModules(self, Spark_devs, SM_devs, VR_devs, haptic_devs):
        print("Starting modules---------------------")
        # modules = [subprocess.Popen(['roscore'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)]
        # time.sleep(1)
        # rospy.init_node('Main', anonymous=True)
        path = os.path.dirname(os.path.abspath(__file__))

        # modules.append(subprocess.Popen(['python3', 'Force/ForceNode.py'], start_new_session=True))
        # modules.append(subprocess.Popen(['python3', 'camera/realsense.py', '/both/front/'], start_new_session=True))
        # modules.append(subprocess.Popen(['python3', 'camera/realsense.py', '/lightning/wrist/'], start_new_session=True))
        # modules.append(subprocess.Popen(['python3', 'camera/realsense.py', '/thunder/wrist/'], start_new_session=True))
        # modules.append(subprocess.Popen(['python3', 'camera/realsense.py', '/both/top/'], start_new_session=True))
        # modules.append(subprocess.Popen(['python3', os.path.join(path, 'Spark/SparkNode_buffer.py')], start_new_session=True))

        modules = []
        for dev in haptic_devs:
            modules.append(subprocess.Popen(['python3', os.path.join(path, 'Haptic/HapticNode.py'), dev], start_new_session=True))
        time.sleep(8)
        print("Modules started----------------------")
        return modules


    def main(self):
        Spark_devs, SM_devs, VR_devs, haptic_devs = self.get_devs()
        modules = self.StartModules(Spark_devs, SM_devs, VR_devs, haptic_devs)
        atexit.register(self.cleanup, modules, None)
        rclpy.spin(self)

if __name__ == '__main__':
    rclpy.init()
    node = LaunchDevs()
    node.main()
    node.destroy_node()
    rclpy.shutdown()
