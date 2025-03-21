# import rospy
import rclpy
import threading
# import torch
import time


class UR5eForceControl:
    def __init__(self, arms):
        self.threads = {}
        self.spark_angle = {}
        self.spark_enable = False
        self.arm_enable = {}
        self.arms = arms
        self.ur5e_DH = [
            [0.0, 0.0, 0.1625, 3.14159/2],
            [0.0, -0.425, 0.0, 0.0],
            [0.0, -0.3922, 0.0, 0.0],
            [0.0, 0.0, 0.1333, 3.14159/2],
            [0.0, 0.0, 0.0997, -3.14159/2],
            [0.0, 0.0, 0.0996, 0.0]
        ]
        self.command_buffer = []

    def set_spark_angle(self, arm, angle):
        self.spark_angle[arm] = angle

    def set_enable(self, enable):
        self.spark_enable = enable

    def start_ik_thread(self, arm):
        self.arm_enable[arm] = True
        if arm not in self.threads:
            self.threads[arm] = threading.Thread(target=self.run_ik_thead, args=(arm,))
            self.threads[arm].start()

    def end_ik_thread(self, arm):
        self.arm_enable[arm] = False
        for arm in self.threads:
            self.threads[arm].join()
        del self.threads[arm]


    def run_ik_thead(self, arm):
        import torch

        # node = rclpy.create_node('ik_thread')


        # rate = rospy.Rate(200)
        # rate = rclpy.Rate(200)
        rate = 200
        while rclpy.ok() and self.arm_enable[arm]:
            start = time.time()
            theta = torch.tensor(self.arms.getActualQ(arm), requires_grad=True)
            total_torque_loss = torch.zeros_like(theta, requires_grad=True)
            ik_loss = torch.zeros(1, requires_grad=True)
            raw_torques = self.arms.getJointTorques(arm)
            torques = raw_torques
            min_torque = 12
            torques = [t-(min_torque*(-1 if t < 0 else 1)) if abs(t) > min_torque else 0 for t in torques]
            targets = torch.tensor(torques)/10000
            targets += theta

            with torch.enable_grad():
                total_torque_loss = torch.sum(torch.abs(theta-targets)**2)

                T = torch.eye(4, requires_grad=True)
                for j in range(len(theta)):
                    a = torch.tensor(self.ur5e_DH[j][1], requires_grad=True)
                    d = torch.tensor(self.ur5e_DH[j][2], requires_grad=True)
                    alpha = torch.tensor(self.ur5e_DH[j][3], requires_grad=True)
                    angle = theta[j]
                    A = torch.stack([
                        torch.stack([torch.cos(angle), -torch.sin(angle) * torch.cos(alpha), torch.sin(angle) * torch.sin(alpha), a * torch.cos(angle)]),
                        torch.stack([torch.sin(angle), torch.cos(angle) * torch.cos(alpha), -torch.cos(angle) * torch.sin(alpha), a * torch.sin(angle)]),
                        torch.stack([torch.tensor(0.0), torch.sin(alpha), torch.cos(alpha), d]),
                        torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)])
                    ])
                    T = T @ A
                ik_loss = torch.abs(T[2, 3] - 0.6) ** 2 * 100

                if self.spark_enable: # SPARK CONTROL
                    spark = torch.tensor(self.spark_angle[arm], requires_grad=True)
                    total_spark_loss = torch.sum(torch.abs(theta-spark[:6])**2)*2
                else:
                    total_spark_loss = torch.tensor(0.0, requires_grad=True)

                max_torque = 40
                maximum_joint_torque = torch.max(torch.abs(torch.tensor(raw_torques)))
                torque = torch.clamp((maximum_joint_torque - min_torque) / (max_torque - min_torque), 0, 0.99)
                spark = 1 - torque
                # print(f"Torque: {torque} Spark: {spark}")
                # print(f"Raw Torques: {raw_torques}")
                loss = total_spark_loss*spark + total_torque_loss*torque

            if self.spark_enable:
                with torch.no_grad():
                    loss.backward()
                    grad = theta.grad
                    grad = (-grad * 1).numpy().tolist()
                    buf_size = 1
                    self.command_buffer.append(grad)
                    if len(self.command_buffer) > buf_size:
                        self.command_buffer.pop(0)
                    command = torch.tensor(self.command_buffer).mean(dim=0).numpy().tolist()
                    # print(f"Command: {command}")
                    # self.control.speedJ(command, 6, 0.005)
                    self.arms.speedJ(arm, [command, 3, 0.001])
            else:
                self.arms.stop(arm)
            # rclpy.spin_once(node)
            duration = time.time() - start
            if duration < 1/rate:
                time.sleep(1/rate - duration)

            # rate.sleep()

