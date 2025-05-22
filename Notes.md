
# how to start
1. start thunder and lightning
2. check spark usb
3. check camera usb
4. conda activate ros_env
5. python TeleopSoftware/launch_devs.py
6. python TeleopSoftware/launch.py
7. change path and language_instruction in data_collection.py
8. run data_collection.py
9. send the data to server in /data folder (e.g. scp -r /home/hanchen/projects/SPARK-Remote/real_world_data/pick_green_into_bowl cui00191@128.101.106.67:/data/cui00191/real_world_data/)


# Todo
1. rclpy.spin_once only receive one topic, so now I run 3 times.
    consider to use multithread? one thread for keyboard and another thread for rclpy.spin()? 
2. remove wrist camera, it blocks the gripper.