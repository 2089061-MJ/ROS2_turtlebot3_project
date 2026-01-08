# 아직까지는 A* + apf 방식이 주행이 잘됨. 
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan

from math import atan2, sqrt, sin, cos, pi
import heapq
import numpy as np
import yaml 
import time

# A* 알고리즘 노드 클래스
class NodeAStar:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0; self.h = 0; self.f = 0
    def __lt__(self, other):
        return self.f < other.f

class RealNavigation(Node):
    def __init__(self):
        super().__init__('integrated_navigation')

        # yaml 파일 불러오기 (시작 좌표를 확인하기 위해서) 
        yaml_path = '/home/teamone/team1_project/src/ros2_project/config/setup.yaml'
        self.start_cfg = {'x': 0.0, 'y': 0.0, 'yaw': -1.57}
        
        try:
            with open(yaml_path, 'r') as f:
                cfg = yaml.safe_load(f)
                if 'start' in cfg: self.start_cfg = cfg['start']
        except Exception: 
            self.get_logger().warn("YAML 로드 실패. (0,0) 시작")

        # 주행 파라미터
        self.MAX_SPEED = 0.15       
        self.LOOK_AHEAD = 0.5       # 코너링 핵심
        self.GOAL_TOL = 0.15        

        self.ATT_GAIN = 1.5         
        # 벽 반발력을 살짝 키움 (기존 0.15 -> 0.20)
        self.REP_GAIN = 0.20        
        self.base_obs_dist = 0.35   

        self.ROBOT_RADIUS = 0.18    
        self.SAFE_MARGIN = 0.03     

        # 변수 초기화
        self.map_data = None; self.curr_pose = None; self.curr_yaw = 0.0
        self.global_path = []; self.path_idx = 0
        self.map_info = {'res':0.05, 'w':0, 'h':0, 'ox':0, 'oy':0}
        
        self.front_dist = 99.9
        self.left_dist = 99.9
        self.right_dist = 99.9
        self.scan_ranges = []

        # 상태 플래그
        self.amcl_synced = False    

        # ros2 pub
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/planned_path', 10)
        self.pub_init = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        # ros2 sub
        self.create_subscription(OccupancyGrid, '/map', self.cb_map, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.cb_pose, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.cb_goal, 10)
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10, 
                         durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(LaserScan, '/scan', self.cb_scan, qos)

        # 제어 루프
        self.create_timer(0.1, self.control_loop)
        
        # AMCL 동기화 체크 타이머
        self.create_timer(0.5, self.check_amcl_sync)

        self.get_logger().info("실전 내비게이션 준비 완료 (충돌 방지 강화)")

    # AMCL 동기화 로직
    def check_amcl_sync(self):
        if self.amcl_synced: return

        target_x = float(self.start_cfg['x'])
        target_y = float(self.start_cfg['y'])
        target_yaw = float(self.start_cfg['yaw'])

        if self.curr_pose is None:
            self.publish_initial_pose()
            return

        # 위치 및 각도 오차 계산
        dist_err = sqrt((self.curr_pose[0] - target_x)**2 + (self.curr_pose[1] - target_y)**2)
        yaw_err = abs(self.curr_yaw - target_yaw)
        while yaw_err > pi: yaw_err -= 2*pi
        yaw_err = abs(yaw_err)

        if dist_err < 0.2 and yaw_err < 0.5:
            self.amcl_synced = True
            self.get_logger().info(f"동기화 완료! (위치오차:{dist_err:.2f}, 각도오차:{yaw_err:.2f})")
            self.get_logger().info("명령 대기 중...")
        else:
            self.publish_initial_pose()
            if self.pub_init.get_subscription_count() > 0:
                self.get_logger().warn(f"위치 보정 중... (Target Yaw: {target_yaw:.2f})")

    def publish_initial_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        
        yaw = float(self.start_cfg['yaw'])
        msg.pose.pose.position.x = float(self.start_cfg['x'])
        msg.pose.pose.position.y = float(self.start_cfg['y'])
        msg.pose.pose.orientation.z = sin(yaw / 2.0)
        msg.pose.pose.orientation.w = cos(yaw / 2.0)
        
        msg.pose.covariance = [0.0]*36
        msg.pose.covariance[0]=0.02; msg.pose.covariance[7]=0.02; msg.pose.covariance[35]=0.01
        
        self.pub_init.publish(msg)

    # 메인 제어 루프
    def control_loop(self):
        if not self.amcl_synced: return
        if not self.global_path or self.curr_pose is None: return

        # 도착 확인
        goal = self.global_path[-1]
        dist_total = sqrt((goal[0]-self.curr_pose[0])**2 + (goal[1]-self.curr_pose[1])**2)
        if dist_total < self.GOAL_TOL:
            self.stop(); self.global_path = []; self.get_logger().info("도착!")
            return

        # 비상 정지 거리 증가 (0.15 -> 0.18)
        if self.front_dist < 0.18:
            self.stop_emergency()
            return

        # APF 계산
        local_goal = self.get_local_goal()
        
        dx = local_goal[0] - self.curr_pose[0]
        dy = local_goal[1] - self.curr_pose[1]
        lx = dx * cos(self.curr_yaw) + dy * sin(self.curr_yaw)
        ly = -dx * sin(self.curr_yaw) + dy * cos(self.curr_yaw)
        
        ld = sqrt(lx**2 + ly**2)
        if ld > 0:
            f_att_x = (lx/ld) * self.ATT_GAIN
            f_att_y = (ly/ld) * self.ATT_GAIN
        else: f_att_x, f_att_y = 0,0

        # 척력
        f_rep_x, f_rep_y = 0.0, 0.0
        if self.left_dist < self.base_obs_dist:
            force = self.REP_GAIN * (1.0/self.left_dist - 1.0/self.base_obs_dist)
            f_rep_y -= force 
        if self.right_dist < self.base_obs_dist:
            force = self.REP_GAIN * (1.0/self.right_dist - 1.0/self.base_obs_dist)
            f_rep_y += force 
        if self.front_dist < self.base_obs_dist:
            force = self.REP_GAIN * (1.0/self.front_dist - 1.0/self.base_obs_dist)
            f_rep_x -= force

        total_x = f_att_x + f_rep_x
        total_y = f_att_y + f_rep_y
        target_ang = atan2(total_y, total_x)

        # 모터 제어 명령
        cmd = Twist()
        
        # 회전: 데드밴드 없이 부드럽게
        cmd.angular.z = target_ang * 1.0
        cmd.angular.z = max(min(cmd.angular.z, 0.8), -0.8)

        # 직진: 90도 이내면 감속하며 전진
        if abs(target_ang) < pi/2:
            cmd.linear.x = self.MAX_SPEED * (1.0 - abs(target_ang)/(pi/2))
        else:
            cmd.linear.x = 0.0

        self.pub_cmd.publish(cmd)

    # 센서 및 유틸리티 
    def cb_scan(self, msg):
        self.scan_ranges = msg.ranges
        count = len(msg.ranges)
        if count > 0:
            # 전방 감지 각도 확대 (기존 +-20도 -> 수정 후 +-40도)
            # 대각선 앞 벽을 '앞'으로 인식하여 미리 멈추거나 피하게 함
            front_range = msg.ranges[0:40] + msg.ranges[-40:]
            
            # 나머지는 좌우
            left_range = msg.ranges[40:90]
            right_range = msg.ranges[-90:-40]

            self.front_dist = self.get_min(front_range)
            self.left_dist = self.get_min(left_range)
            self.right_dist = self.get_min(right_range)

    def get_min(self, ranges):
        v = [r for r in ranges if 0.05 < r < 10.0]
        return min(v) if v else 99.9

    def stop_emergency(self):
        cmd = Twist()
        cmd.linear.x = -0.05; cmd.angular.z = 0.0
        self.pub_cmd.publish(cmd)

    def cb_goal(self, msg):
        if not self.amcl_synced: 
            self.get_logger().warn("AMCL 동기화 대기 중...")
            return
        
        sx, sy = self.w2g(self.curr_pose)
        gx, gy = self.w2g([msg.pose.position.x, msg.pose.position.y])
        path = self.run_astar((sy, sx), (gy, gx))
        if path:
            self.global_path = [[p[1]*self.map_info['res']+self.map_info['ox'], 
                                 p[0]*self.map_info['res']+self.map_info['oy']] for p in path]
            self.path_idx = 0; self.viz_path(); self.get_logger().info("출발!")

    def run_astar(self, start, end):
        start_node = NodeAStar(None, start)
        open_list = []; heapq.heappush(open_list, start_node)
        visited = set()
        moves = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        while open_list:
            cur = heapq.heappop(open_list)
            if cur.position in visited: continue
            visited.add(cur.position)
            if cur.position == end:
                path = []; 
                while cur: path.append(cur.position); cur = cur.parent
                return path[::-1]
            for dy, dx in moves:
                ny, nx = cur.position[0]+dy, cur.position[1]+dx
                if not (0<=ny<self.map_info['h'] and 0<=nx<self.map_info['w']): continue
                if not self.is_safe(ny, nx): continue
                node = NodeAStar(cur, (ny, nx))
                node.g = cur.g+1
                node.h = sqrt((ny-end[0])**2+(nx-end[1])**2)
                node.f = node.g+node.h
                heapq.heappush(open_list, node)
        return None

    def get_local_goal(self):
        idx = self.path_idx
        for i in range(self.path_idx, len(self.global_path)):
            p = self.global_path[i]
            d = sqrt((p[0]-self.curr_pose[0])**2 + (p[1]-self.curr_pose[1])**2)
            if d >= self.LOOK_AHEAD: idx = i; break
        self.path_idx = idx
        return self.global_path[idx]

    def is_safe(self, y, x):
        steps = int((self.ROBOT_RADIUS + self.SAFE_MARGIN) / self.map_info['res'])
        for dy in range(-steps, steps+1):
            for dx in range(-steps, steps+1):
                ny, nx = y+dy, x+dx
                if 0<=ny<self.map_info['h'] and 0<=nx<self.map_info['w']:
                    if self.map_data[ny][nx] != 0: return False
        return True

    def cb_map(self, msg):
        self.map_info['res'] = msg.info.resolution
        self.map_info['w'] = msg.info.width
        self.map_info['h'] = msg.info.height
        self.map_info['ox'] = msg.info.origin.position.x
        self.map_info['oy'] = msg.info.origin.position.y
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))

    def cb_pose(self, msg):
        self.curr_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        q = msg.pose.pose.orientation
        self.curr_yaw = atan2(2.0*(q.w*q.z+q.x*q.y), 1.0-2.0*(q.y*q.y+q.z*q.z))

    def w2g(self, p):
        return int((p[0]-self.map_info['ox'])/self.map_info['res']), int((p[1]-self.map_info['oy'])/self.map_info['res'])
    
    def viz_path(self):
        msg = Path(); msg.header.frame_id = 'map'
        for p in self.global_path:
            ps = PoseStamped(); ps.pose.position.x = p[0]; ps.pose.position.y = p[1]
            msg.poses.append(ps)
        self.pub_path.publish(msg)
    def stop(self): self.pub_cmd.publish(Twist())

def main(args=None):
    rclpy.init(args=args); node = RealNavigation()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.stop(); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()