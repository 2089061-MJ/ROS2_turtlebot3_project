# 목적지 도달하고서 다른 지점으로 이동할 때 장애물 인식 거리 때문에 회전을 안하고 벽에 갇힌 오류가 발생했으나 수정함.
# 주행 파라미터도 미세하게 조정 (이제 벽에 바짝 붙어서 이동하지 않습니다.)
# 동적 장애물 감지할 때 정지 기능은 구현되어있습니다. 회피 이동은 별도로 구현이 필요합니다.
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

        # yaml 파일 불러오기
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
        self.LOOK_AHEAD = 0.5       
        self.GOAL_TOL = 0.15        

        self.ATT_GAIN = 1.5         
        # 벽에서 밀어내는 힘 강화 (0.2 -> 0.3)
        self.REP_GAIN = 0.30        
        # 벽 감지 시작 거리 증가 (0.35m -> 0.50m) -> 미리 피함
        self.base_obs_dist = 0.50   

        self.ROBOT_RADIUS = 0.18    
        # 경로 생성 여유폭 대폭 증가 (0.03m -> 0.12m)
        # 이제 A* 경로 자체가 벽에서 12cm 더 떨어져서 생성됩니다.
        self.SAFE_MARGIN = 0.12     

        # 비상 정지 기준
        self.NORMAL_STOP_DIST = 0.18    # 평소 주행 시 정지 거리 (벽)
        self.ROTATE_STOP_DIST = 0.11    # 회전 시 허용 거리 (벽)
        
        self.OBSTACLE_STOP_DIST = 0.35  # 전방 장애물(동적) 감지 거리
        

        # 변수 초기화
        self.map_data = None; self.curr_pose = None; self.curr_yaw = 0.0
        self.global_path = []; self.path_idx = 0
        self.map_info = {'res':0.05, 'w':0, 'h':0, 'ox':0, 'oy':0}
        
        # 거리 변수 초기화
        self.front_dist = 99.9      # 벽 감지용
        self.long_front_dist = 99.9 # 장애물 감지용
        self.left_dist = 99.9
        self.right_dist = 99.9
        self.scan_ranges = []

        self.amcl_synced = False    

        # 통신 설정 ros2 pub
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

        self.create_timer(0.1, self.control_loop)
        self.create_timer(0.5, self.check_amcl_sync)

        self.get_logger().info("통합 내비게이션 노드 실행중입니다.")

    
    # 센서 데이터 분리
    def cb_scan(self, msg):
        self.scan_ranges = msg.ranges
        count = len(msg.ranges)
        if count > 0:
            wide_range = msg.ranges[0:40] + msg.ranges[-40:]
            self.front_dist = self.get_min(wide_range)

            narrow_range = msg.ranges[0:10] + msg.ranges[-10:]
            self.long_front_dist = self.get_min(narrow_range)

            self.left_dist = self.get_min(msg.ranges[40:90])
            self.right_dist = self.get_min(msg.ranges[-90:-40])

    def get_min(self, ranges):
        v = [r for r in ranges if 0.05 < r < 10.0]
        return min(v) if v else 99.9

    # 제어 루프
    def control_loop(self):
        if not self.amcl_synced: return
        if not self.global_path or self.curr_pose is None: return

        # 1. 도착 확인
        goal = self.global_path[-1]
        dist_total = sqrt((goal[0]-self.curr_pose[0])**2 + (goal[1]-self.curr_pose[1])**2)
        if dist_total < self.GOAL_TOL:
            self.stop(); self.global_path = []; self.get_logger().info("도착!")
            return

        # 목표 각도 계산
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

        # 상황별 정지 조건
        is_turning = abs(target_ang) > 0.7  
        
        if is_turning:
            limit_dist = self.ROTATE_STOP_DIST
        else:
            limit_dist = self.NORMAL_STOP_DIST

        # (A) 벽 충돌 체크
        if self.front_dist < limit_dist:
            self.stop_emergency()
            self.get_logger().warn(f"벽 너무 가까움({self.front_dist:.2f}m)! 정지")
            return

        # (B) 전방 장애물 체크 (회전 시 무시)
        if not is_turning and self.long_front_dist < self.OBSTACLE_STOP_DIST:
            self.stop_emergency()
            self.get_logger().warn(f"전방 장애물 발견! 정지")
            return

        # 모터 명령
        cmd = Twist()
        cmd.angular.z = target_ang * 1.0
        cmd.angular.z = max(min(cmd.angular.z, 0.8), -0.8)

        if abs(target_ang) < pi/2:
            cmd.linear.x = self.MAX_SPEED * (1.0 - abs(target_ang)/(pi/2))
        else:
            cmd.linear.x = 0.0

        self.pub_cmd.publish(cmd)

    def stop(self): self.pub_cmd.publish(Twist())
    def stop_emergency(self):
        cmd = Twist()
        cmd.linear.x = 0.0; cmd.angular.z = 0.0
        self.pub_cmd.publish(cmd)

    # 유틸리티
    def check_amcl_sync(self):
        if self.amcl_synced: return
        target_x = float(self.start_cfg['x'])
        target_y = float(self.start_cfg['y'])
        if self.curr_pose is None: self.publish_initial_pose(); return
        dist = sqrt((self.curr_pose[0]-target_x)**2 + (self.curr_pose[1]-target_y)**2)
        if dist < 0.2: 
            self.amcl_synced = True; self.get_logger().info("동기화 완료!")
        else: self.publish_initial_pose()

    def publish_initial_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'; msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(self.start_cfg['x'])
        msg.pose.pose.position.y = float(self.start_cfg['y'])
        msg.pose.pose.orientation.z = sin(float(self.start_cfg['yaw'])/2)
        msg.pose.pose.orientation.w = cos(float(self.start_cfg['yaw'])/2)
        msg.pose.covariance = [0.0]*36; msg.pose.covariance[0]=0.02; msg.pose.covariance[35]=0.01
        self.pub_init.publish(msg)

    def cb_goal(self, msg):
        if not self.amcl_synced: return
        sx, sy = self.w2g(self.curr_pose)
        gx, gy = self.w2g([msg.pose.position.x, msg.pose.position.y])
        path = self.run_astar((sy, sx), (gy, gx))
        if path:
            self.global_path = [[p[1]*self.map_info['res']+self.map_info['ox'], 
                                 p[0]*self.map_info['res']+self.map_info['oy']] for p in path]
            self.path_idx = 0; self.viz_path(); self.get_logger().info("새로운 목적지 Goal로 이동합니다. 출발")

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

def main(args=None):
    rclpy.init(args=args); node = RealNavigation()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.stop(); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()