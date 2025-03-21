import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from datetime import datetime, timedelta

class TopicChecker(Node):
    def __init__(self, topics, timeout_ms=500):
        super().__init__('topic_checker')
        self.topics = topics
        self.timeout = timedelta(milliseconds=timeout_ms)
        self.last_published_times = {topic: datetime.min for topic in topics}
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )
        
        self.subscribers = []
        for topic in topics:
            self.subscribers.append(self.create_subscription(
                Image,
                topic,
                lambda msg, t=topic: self.topic_callback(msg, t),
                qos_profile
            ))
        
        self.timer = self.create_timer(5, self.check_topics)

    def topic_callback(self, msg, topic):
        self.last_published_times[topic] = datetime.now()

    def check_topics(self):
        now = datetime.now()
        all_good = True
        for topic, last_time in self.last_published_times.items():
            if now - last_time > self.timeout:
                self.get_logger().warn(f"Topic {topic} has not been published in the last {self.timeout.total_seconds()} seconds")
                all_good = False
        if all_good:
            self.get_logger().info(f"All topics are good")
def main(args=None):
    rclpy.init(args=args)
    topics = ['/video_frames']
    node = TopicChecker(topics)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()