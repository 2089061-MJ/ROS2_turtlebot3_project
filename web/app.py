from flask import Flask, render_template, request, jsonify
import yaml
import rclpy
from geometry_msgs.msg import PoseStamped
import threading

app = Flask(__name__)

# YAML 파일
with open('/home/teamone/team1_project/src/ros2_project/config/setup.yaml') as f:
    DEST_DATA = yaml.safe_load(f)

# ROS2 노드 설정
rclpy.init()
node = rclpy.create_node('web_goal_publisher')
pub_goal = node.create_publisher(PoseStamped, '/goal_pose', 10)

def ros_spin():
    rclpy.spin(node)

threading.Thread(target=ros_spin, daemon=True).start()

def send_ros_goal(x, y, yaw):
    msg = PoseStamped()
    msg.header.frame_id = 'map'
    msg.pose.position.x = x
    msg.pose.position.y = y
    # yaw → quaternion 변환
    import math
    import tf_transformations
    q = tf_transformations.quaternion_from_euler(0, 0, yaw)
    msg.pose.orientation.x = q[0]
    msg.pose.orientation.y = q[1]
    msg.pose.orientation.z = q[2]
    msg.pose.orientation.w = q[3]
    pub_goal.publish(msg)

@app.route('/')
def index():
    return render_template('index.html',
                           start=DEST_DATA['start'],
                           destinations=DEST_DATA['destinations'])

@app.route('/api/goal', methods=['POST'])
def api_goal():
    name = request.json['name']
    if name == DEST_DATA['start']['name']:
        goal = DEST_DATA['start']
    else:
        goal = next(d for d in DEST_DATA['destinations'] if d['name'] == name)
    send_ros_goal(goal['x'], goal['y'], goal['yaw'])
    return jsonify({"status":"goal_sent","goal":name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
