#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, Int32
import time
import pickle
import os
import sys
import termios
import tty
import select
import cv2

from cv_bridge import CvBridge
from camera import RealSenseCamera



# data collection settings
save_dir = "./real_world_data/test/"
LANG_INSTRUCTION = "pick the green block into the black bowl."
os.makedirs(save_dir, exist_ok=True)
step_hz = 10
step_dt = 1.0 / step_hz


# ========== function：Non-blocking keyboard input ==========
def get_key():
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None

def set_cbreak_mode():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return old_settings

def restore_terminal_mode(old_settings):
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ========== ROS2 StateSubscriber ==========
class StateSubscriber(Node):
    def __init__(self):
        super().__init__('lightning_state_subscriber')

        self.eef_pose = None
        self.joint_positions = None
        self.gripper_value = None

        self.create_subscription(Float32MultiArray, '/lightning_cartesian_eef', self.eef_callback, 10)
        self.create_subscription(Float32MultiArray, '/lightning_q', self.q_callback, 10)
        self.create_subscription(Int32, '/lightning_gripper', self.gripper_callback, 10)

    def eef_callback(self, msg):
        self.eef_pose = msg.data

    def q_callback(self, msg):
        self.joint_positions = msg.data

    def gripper_callback(self, msg):
        self.gripper_value = msg.data


# ========== main function ==========

rclpy.init()
state_sub = StateSubscriber()
cam = RealSenseCamera()


def main():
    recording = False
    frames = []
    traj_id = int(time.time())

    print("Press 's' to start recording, 'e' to end recording, 'q' to quit.")

    last_step_time = time.time()
    old_terminal_settings = set_cbreak_mode()

    try:
        while True:
            key = get_key()

            if key == 's' and not recording:
                print("[INFO] Start recording...")
                recording = True
                frames = []
                time.sleep(0.2)

            elif key == 'e' and recording:
                print("[INFO] Stop recording. Saving...")
                save_trajectory(frames, traj_id)
                traj_id = int(time.time())
                recording = False
                time.sleep(0.2)
                print("[INFO] Number of trajectories collected:", len(os.listdir(save_dir)))

            elif key == 'q':
                print("[INFO] Quit program.")
                break

            now = time.time()
            if recording and (now - last_step_time) >= step_dt:
                rclpy.spin_once(state_sub, timeout_sec=0.01)
                frame = collect_one_frame()
                if frame is not None:
                    frames.append(frame)
                last_step_time = now

    finally:
        restore_terminal_mode(old_terminal_settings)
        cam.stop()
        state_sub.destroy_node()
        rclpy.shutdown()


def collect_one_frame():
    if (
        state_sub.eef_pose is None or
        state_sub.joint_positions is None or
        state_sub.gripper_value is None
    ):
        return None

    color_image = cam.get_color_frame()  # BGR
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    timestamp = time.time()
    frame = {
        "timestamp": timestamp,
        "rgb": color_image,
        "joint_positions": state_sub.joint_positions,
        "eef_pose": {
            "position": state_sub.eef_pose[:3],
            "orientation_rpy": state_sub.eef_pose[3:]
        },
        "gripper_state": state_sub.gripper_value,
        "lang_instruction": LANG_INSTRUCTION,
    }
    return frame


def save_trajectory(frames, traj_id):
    traj = {
        "meta": {
            "traj_id": traj_id,
            "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(traj_id)),
        },
        "frames": frames
    }
    save_path = os.path.join(save_dir, f"traj_{traj_id}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Trajectory saved: {save_path}")


if __name__ == "__main__":
    main()
