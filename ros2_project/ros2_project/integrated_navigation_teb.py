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
import copy

# [1] A* 노드 클래스
class NodeAStar:
    def __init__(self, parent=None, position=None):
        self.parent = parent; self.position = position
        self.g = 0; self.h = 0; self.f = 0
    def __lt__(self, other): return self.f < other.f

class RealNavigation(Node):
    def __init__(self):
        super().__init__('integrated_navigation')
        
        # 설정 파일 로드
        self.start_cfg = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        try:
            with open('/home/teamone/team1_project/src/ros2_project/config/setup.yaml', 'r') as f:
                cfg = yaml.safe_load(f)
                if 'start' in cfg: self.start_cfg = cfg['start']
        except: pass

        # [2] 주행 및 최적화 파라미터
        self.MAX_SPEED = 0.15       # 최고 속도
        self.LOOK_AHEAD = 0.6       # 경로 추종 거리 (부드러운 주행을 위해 약간 길게)
        self.GOAL_TOL = 0.15        # 도착 판정 거리
        
        # [TEB Elastic Band 파라미터]
        self.EB_ALPHA = 0.1         # 경로를 팽팽하게 당기는 힘 (스무딩)
        self.EB_BETA = 0.3          # 장애물에서 밀어내는 힘 (회피)
        self.EB_ITERATIONS = 50     # 최적화 반복 횟수

        # [안전 파라미터]
        self.ROBOT_RADIUS = 0.18
        self.SAFE_MARGIN = 0.03
        self.OBS_DIST = 0.35        # 주행 중 벽 회피(Nudge) 민감도

        # 변수 초기화
        self.curr_pose = None; self.curr_yaw = 0.0
        self.global_path = []; self.path_idx = 0
        self.map_data = None; self.map_info = {'res':0.05, 'w':0, 'h':0, 'ox':0, 'oy':0}
        
        self.front_dist = 99.9; self.left_dist = 99.9; self.right_dist = 99.9
        self.amcl_synced = False

        # 통신 설정
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/planned_path', 10)
        self.pub_init = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10, 
                         durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(LaserScan, '/scan', self.cb_scan, qos)
        self.create_subscription(OccupancyGrid, '/map', self.cb_map, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.cb_pose, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.cb_goal, 10)

        self.create_timer(0.1, self.control_loop)
        self.create_timer(0.5, self.check_amcl_sync)
        self.get_logger().info("TEB Lite + 안전 주행 모드 준비 완료")

    # [3] TEB 스타일 경로 최적화 (Elastic Band)
    def optimize_path(self, original_path):
        if not original_path or len(original_path) < 3:
            return original_path

        path = np.array(original_path, dtype=float)
        path_len = len(path)
        
        for _ in range(self.EB_ITERATIONS):
            new_path = np.copy(path)
            
            for i in range(1, path_len - 1):
                # 1. Internal Force: 직선으로 펴지려는 힘
                smoothing_force = (path[i-1] + path[i+1]) / 2.0 - path[i]
                
                # 2. External Force: 장애물에서 멀어지려는 힘
                obs_force = np.array([0.0, 0.0])
                
                px, py = path[i]
                gx = int((px - self.map_info['ox']) / self.map_info['res'])
                gy = int((py - self.map_info['oy']) / self.map_info['res'])
                
                # 주변 탐색 (장애물 찾기)
                min_dist = float('inf')
                nearest_obs = None
                search_range = 4 # 약 20cm 주변 탐색
                found = False
                
                for r in range(-search_range, search_range+1):
                    for c in range(-search_range, search_range+1):
                        ny, nx = gy + r, gx + c
                        if 0 <= ny < self.map_info['h'] and 0 <= nx < self.map_info['w']:
                            if self.map_data[ny][nx] != 0:
                                ox = nx * self.map_info['res'] + self.map_info['ox']
                                oy = ny * self.map_info['res'] + self.map_info['oy']
                                d = sqrt((px-ox)**2 + (py-oy)**2)
                                if d < min_dist:
                                    min_dist = d
                                    nearest_obs = np.array([ox, oy])
                                    found = True
                
                if found and min_dist < 0.3: # 30cm 이내 장애물에 반응
                    push_dir = path[i] - nearest_obs
                    norm = np.linalg.norm(push_dir)
                    if norm > 0:
                        obs_force = (push_dir / norm) * (0.3 - min_dist)

                # 위치 업데이트
                new_path[i] += self.EB_ALPHA * smoothing_force + self.EB_BETA * obs_force

            path = new_path

        return path.tolist()

    # [4] 메인 제어 루프
    def control_loop(self):
        if not self.amcl_synced or not self.global_path or self.curr_pose is None: return

        goal = self.global_path[-1]
        dist_to_goal = sqrt((goal[0]-self.curr_pose[0])**2 + (goal[1]-self.curr_pose[1])**2)

        # 도착 판정
        if dist_to_goal < self.GOAL_TOL:
            self.stop(); self.global_path = []; self.get_logger().info("도착!")
            return

        # [안전] 비상 정지 (전방 18cm)
        if self.front_dist < 0.18:
            self.stop_emergency(); return

        # Pure Pursuit
        local_goal = self.get_local_goal()
        target_yaw = atan2(local_goal[1] - self.curr_pose[1], local_goal[0] - self.curr_pose[0])
        
        err_yaw = target_yaw - self.curr_yaw
        while err_yaw > pi: err_yaw -= 2*pi
        while err_yaw < -pi: err_yaw += 2*pi

        # [회피] Nudge Logic (벽이 가까우면 핸들 꺾기)
        if self.left_dist < self.OBS_DIST: 
            err_yaw -= 0.25  # 오른쪽으로 회피
        elif self.right_dist < self.OBS_DIST: 
            err_yaw += 0.25  # 왼쪽으로 회피

        # 모터 명령
        cmd = Twist()
        # 회전 속도 (최대 1.0 rad/s)
        cmd.angular.z = max(min(err_yaw * 1.5, 1.0), -1.0)
        
        # 속도 조절 (코너 및 도착 시 감속)
        speed = self.MAX_SPEED
        if abs(err_yaw) > 0.5: speed *= 0.5 # 코너링 감속
        if dist_to_goal < 0.5: speed = min(speed, dist_to_goal * 1.0) # 도착 감속
        
        cmd.linear.x = speed
        self.pub_cmd.publish(cmd)

    # ==========================================
    # [5] 센서 및 유틸리티 (중요!)
    # ==========================================
    def cb_scan(self, msg):
        self.scan_ranges = msg.ranges
        count = len(msg.ranges)
        if count > 0:
            # [핵심] 전방 감지각 ±40도 (어깨 충돌 방지)
            front_range = msg.ranges[0:40] + msg.ranges[-40:]
            
            # 측면 감지 (회피용)
            left_range = msg.ranges[40:90]
            right_range = msg.ranges[-90:-40]

            self.front_dist = self.get_min(front_range)
            self.left_dist = self.get_min(left_range)
            self.right_dist = self.get_min(right_range)

    def get_min(self, ranges):
        v = [r for r in ranges if 0.05 < r < 10.0]
        return min(v) if v else 99.9

    def check_amcl_sync(self):
        if self.amcl_synced: return
        target = self.start_cfg
        if self.curr_pose is None: self.pub_init_pose(); return
        dist = sqrt((self.curr_pose[0]-target['x'])**2 + (self.curr_pose[1]-target['y'])**2)
        if dist < 0.2: self.amcl_synced = True; self.get_logger().info("✅ 동기화 완료!")
        else: self.pub_init_pose()

    def pub_init_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'; msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(self.start_cfg['x'])
        msg.pose.pose.position.y = float(self.start_cfg['y'])
        msg.pose.pose.orientation.z = sin(float(self.start_cfg['yaw'])/2)
        msg.pose.pose.orientation.w = cos(float(self.start_cfg['yaw'])/2)
        msg.pose.covariance = [0.0]*36; msg.pose.covariance[0]=0.05; msg.pose.covariance[35]=0.02
        self.pub_init.publish(msg)

    def cb_goal(self, msg):
        if not self.amcl_synced: return
        sx, sy = self.w2g(self.curr_pose)
        gx, gy = self.w2g([msg.pose.position.x, msg.pose.position.y])
        
        # 1. A* 경로 생성
        raw_path = self.run_astar((sy, sx), (gy, gx))
        if raw_path:
            world_path = [[p[1]*self.map_info['res']+self.map_info['ox'], 
                           p[0]*self.map_info['res']+self.map_info['oy']] for p in raw_path]
            
            # 2. TEB 스타일 최적화 (경로 펴기)
            self.get_logger().info("경로 최적화 중...")
            optimized_path = self.optimize_path(world_path)
            
            self.global_path = optimized_path
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
                if self.is_safe(ny, nx):
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
            if sqrt((p[0]-self.curr_pose[0])**2 + (p[1]-self.curr_pose[1])**2) >= self.LOOK_AHEAD:
                idx = i; break
        self.path_idx = idx
        return self.global_path[idx]

    def is_safe(self, y, x):
        if self.map_data[y][x] != 0: return False 
        r = int(self.ROBOT_RADIUS / self.map_info['res'])
        if self.map_data[y+r][x] != 0 or self.map_data[y-r][x] != 0: return False
        if self.map_data[y][x+r] != 0 or self.map_data[y][x-r] != 0: return False
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
    def stop_emergency(self): self.pub_cmd.publish(Twist())

def main(args=None):
    rclpy.init(args=args); node = RealNavigation()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.stop(); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()