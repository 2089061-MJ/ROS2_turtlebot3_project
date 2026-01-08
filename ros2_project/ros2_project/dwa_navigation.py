import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from math import atan2, sqrt, sin, cos, pi, hypot
import heapq
import numpy as np
import yaml

# =================================================
# [0] 설정 클래스 (미로 주행용 튜닝 완료)
# =================================================
class Config:
    def __init__(self):
        # 1. 로봇 물리 정보
        self.max_speed = 0.18       # [m/s] 미로에서는 너무 빠르면 위험 (0.22 -> 0.18)
        self.min_speed = -0.05      # [m/s] 후진은 최소화
        self.max_yaw_rate = 1.0     # [rad/s]
        self.max_accel = 0.2        # [m/ss] 급발진 방지
        self.max_delta_yaw_rate = 1.5 
        
        # 2. 로봇 크기 (중요!)
        # 좁은 미로를 통과하려면 실제보다 약간만 크게 잡아야 함
        self.robot_radius = 0.14    # [m] (TurtleBot3 기준 약 0.12~0.15 권장)

        # 3. DWA 해상도 및 예측 (연산 최적화)
        self.v_resolution = 0.02    # [m/s] 0.01 -> 0.02 (계산량 절반 감소)
        self.yaw_rate_resolution = 0.1 # [rad/s] 0.05 -> 0.1 (계산량 절반 감소)
        self.dt = 0.1               # [s]
        self.predict_time = 1.0     # [s] 2.0 -> 1.0 (좁은 길에서는 먼 미래보다 당장이 중요)
        
        # 4. 비용 함수 가중치
        self.to_goal_cost_gain = 0.30  # 목적지 방향 (길을 잃지 않도록 강화)
        self.speed_cost_gain = 0.5     # 속도 (천천히 가도 됨)
        self.obstacle_cost_gain = 1.2  # 장애물 (충돌은 절대 안 됨)
        self.stuck_flag_cons = 0.001   # 정지 판단 기준

# [1] A* 노드
class NodeAStar:
    def __init__(self, parent=None, pos=None):
        self.parent = parent; self.pos = pos
        self.f = 0; self.g = 0; self.h = 0
    def __lt__(self, other): return self.f < other.f

class SimpleNavigation(Node):
    def __init__(self):
        super().__init__('simple_navigation')
        
        self.config = Config()
        
        # [A* 설정] 벽에서 몇 칸 띄울지 (좁은 미로라면 2칸 추천)
        self.WALL_PADDING = 2 
        self.STOP_DIST = 0.20  # 비상 정지 거리

        # 상태 변수
        # [x, y, yaw, v, w] (로봇 기준 좌표계 시뮬레이션용)
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 
        
        self.is_ready = False
        self.curr_pose = None
        self.curr_yaw = 0.0
        self.global_path = []
        self.path_idx = 0
        self.scan_obs = [] 
        self.front_min_dist = 9.9
        self.map_info = {'res':0.05, 'w':0, 'h':0, 'ox':0, 'oy':0}
        self.map_data = None

        # ROS2 Communication
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/plan', 10)
        self.pub_dwa = self.create_publisher(Path, '/dwa', 10)
        self.pub_init = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10, 
                         durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(LaserScan, '/scan', self.cb_scan, qos)
        self.create_subscription(OccupancyGrid, '/map', self.cb_map, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.cb_pose, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.cb_goal, 10)

        self.create_timer(0.1, self.control_loop)
        self.load_start_pose()
        self.get_logger().info("✅ 미로 최적화 DWA 내비게이션 준비 완료")

    # =================================================
    # [2] 메인 제어 루프
    # =================================================
    def control_loop(self):
        if not self.is_ready: self.check_amcl_sync(); return
        if not self.global_path: return

        # 1. 비상 정지 (물리적 안전장치)
        if self.front_min_dist < self.STOP_DIST:
            self.stop(); self.get_logger().warn("🚨 비상 정지!", throttle_duration_sec=1.0)
            return

        # 2. 도착 확인
        goal_global = self.global_path[-1]
        dist_to_goal = hypot(goal_global[0]-self.curr_pose[0], goal_global[1]-self.curr_pose[1])
        if dist_to_goal < 0.15:
            self.stop(); self.global_path = []; self.get_logger().info("🏁 도착!")
            return

        # 3. Local Goal 좌표 변환 (Global -> Robot Frame)
        local_goal = self.get_local_goal()
        dx = local_goal[0] - self.curr_pose[0]
        dy = local_goal[1] - self.curr_pose[1]
        
        # 회전 행렬을 통해 로봇 기준 좌표(x:전방, y:좌측)로 변환
        gx = dx * cos(self.curr_yaw) + dy * sin(self.curr_yaw)
        gy = -dx * sin(self.curr_yaw) + dy * cos(self.curr_yaw)
        goal_robot = np.array([gx, gy])

        # 4. DWA 실행
        # 시뮬레이션용 상태: x=0, y=0, yaw=0 (로봇 중심), 속도는 현재 속도 유지
        sim_state = np.array([0.0, 0.0, 0.0, self.state[3], self.state[4]])
        u, traj = self.dwa_control(sim_state, goal_robot, self.scan_obs)

        # 5. 이동 및 상태 업데이트
        self.state[3] = u[0]
        self.state[4] = u[1]
        self.move(u[0], u[1])
        self.viz_dwa(traj)

    # =================================================
    # [3] DWA 로직 (PythonRobotics Style)
    # =================================================
    def dwa_control(self, x, goal, ob):
        dw = self.calc_dynamic_window(x)
        u, traj = self.calc_control_and_trajectory(x, dw, goal, ob)
        return u, traj

    def calc_dynamic_window(self, x):
        # 로봇 한계 vs 현재 속도에서 가속 가능한 범위 교집합
        Vs = [self.config.min_speed, self.config.max_speed,
              -self.config.max_yaw_rate, self.config.max_yaw_rate]
        
        Vd = [x[3] - self.config.max_accel * self.config.dt,
              x[3] + self.config.max_accel * self.config.dt,
              x[4] - self.config.max_delta_yaw_rate * self.config.dt,
              x[4] + self.config.max_delta_yaw_rate * self.config.dt]

        return [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    def calc_control_and_trajectory(self, x, dw, goal, ob):
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_traj = np.array(x)

        # 해상도만큼 반복하며 최적 속도 탐색
        for v in np.arange(dw[0], dw[1], self.config.v_resolution):
            for w in np.arange(dw[2], dw[3], self.config.yaw_rate_resolution):
                
                traj = self.predict_trajectory(x, v, w)

                # 비용 계산
                to_goal = self.config.to_goal_cost_gain * self.calc_to_goal_cost(traj, goal)
                speed = self.config.speed_cost_gain * (self.config.max_speed - traj[-1, 3])
                ob_cost = self.config.obstacle_cost_gain * self.calc_obstacle_cost(traj, ob)

                final_cost = to_goal + speed + ob_cost

                if final_cost < min_cost:
                    min_cost = final_cost
                    best_u = [v, w]
                    best_traj = traj
                    
                    # 충돌 경로는 즉시 0.0 속도로 대체 (방어 코드)
                    if ob_cost == float("inf"):
                        best_u = [0.0, 0.0]

        return best_u, best_traj

    def predict_trajectory(self, x_init, v, w):
        x = np.array(x_init)
        traj = np.array(x)
        time = 0
        while time <= self.config.predict_time:
            x[2] += w * self.config.dt
            x[0] += v * cos(x[2]) * self.config.dt
            x[1] += v * sin(x[2]) * self.config.dt
            x[3] = v; x[4] = w
            traj = np.vstack((traj, x))
            time += self.config.dt
        return traj

    def calc_to_goal_cost(self, traj, goal):
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        error_angle = atan2(dy, dx)
        cost_angle = error_angle - traj[-1, 2]
        cost = abs(atan2(sin(cost_angle), cos(cost_angle)))
        return cost

    def calc_obstacle_cost(self, traj, ob):
        if not ob: return 0.0
        
        # [최적화] Numpy 연산으로 속도 향상
        ox = np.array([o[0] for o in ob])
        oy = np.array([o[1] for o in ob])
        
        # [최적화] 경로의 모든 점을 다 검사하지 않고 3개씩 건너뜀 (속도 향상)
        traj_downsampled = traj[::3] 
        
        min_r = float("inf")
        for pt in traj_downsampled:
            dx = pt[0] - ox
            dy = pt[1] - oy
            d = np.hypot(dx, dy)
            current_min = np.min(d)
            
            if current_min <= self.config.robot_radius:
                return float("inf")
            min_r = min(min_r, current_min)
            
        return 1.0 / min_r if min_r > 0 else float("inf")

    # =================================================
    # [4] A* (전역 경로 생성) - 안전 마진 적용
    # =================================================
    def run_astar(self, start, end):
        start_node = NodeAStar(None, start)
        open_list = []; heapq.heappush(open_list, start_node)
        visited = set()
        moves = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

        while open_list:
            cur = heapq.heappop(open_list)
            if cur.pos in visited: continue
            visited.add(cur.pos)

            if cur.pos == end:
                path = []
                while cur: path.append(cur.pos); cur = cur.parent
                return path[::-1]

            for dy, dx in moves:
                ny, nx = cur.pos[0]+dy, cur.pos[1]+dx
                if not (0<=ny<self.map_info['h'] and 0<=nx<self.map_info['w']): continue
                
                # [안전] 벽 팽창 확인
                if not self.check_safe_cell(ny, nx): continue

                node = NodeAStar(cur, (ny, nx))
                node.g = cur.g + 1
                node.h = hypot(ny-end[0], nx-end[1])
                node.f = node.g + node.h
                heapq.heappush(open_list, node)
        return None

    def check_safe_cell(self, r, c):
        pad = self.WALL_PADDING
        # 중심 체크
        if self.map_data[r][c] != 0: return False
        # 주변 팽창 체크
        for i in range(r-pad, r+pad+1):
             for j in range(c-pad, c+pad+1):
                 if 0<=i<self.map_info['h'] and 0<=j<self.map_info['w']:
                     if self.map_data[i][j] != 0: return False
        return True

    # =================================================
    # [5] 센서 및 유틸리티
    # =================================================
    def cb_scan(self, msg):
        # [최적화] 중요 데이터만 추출
        angle_inc = msg.angle_increment
        obs = []
        
        # 1. 비상 정지용 전방 거리
        front_ranges = msg.ranges[0:30] + msg.ranges[-30:]
        v_front = [r for r in front_ranges if 0.05 < r < 5.0]
        self.front_min_dist = min(v_front) if v_front else 9.9
        
        # 2. DWA용 장애물 (다운샘플링 & 거리제한)
        # 4개씩 건너뛰기, 1.5m 이내만 (미로라서 멀리 볼 필요 X)
        for i, r in enumerate(msg.ranges):
            if i % 4 == 0 and 0.05 < r < 1.5: 
                theta = msg.angle_min + i * angle_inc
                obs.append([r * cos(theta), r * sin(theta)])
        self.scan_obs = obs

    def cb_goal(self, msg):
        if not self.is_ready: return
        sx = int((self.curr_pose[0]-self.map_info['ox'])/self.map_info['res'])
        sy = int((self.curr_pose[1]-self.map_info['oy'])/self.map_info['res'])
        gx = int((msg.pose.position.x-self.map_info['ox'])/self.map_info['res'])
        gy = int((msg.pose.position.y-self.map_info['oy'])/self.map_info['res'])

        self.get_logger().info("경로 탐색 시작...")
        path = self.run_astar((sy, sx), (gy, gx))
        if path:
            self.global_path = [[p[1]*self.map_info['res']+self.map_info['ox'], 
                                 p[0]*self.map_info['res']+self.map_info['oy']] for p in path]
            self.path_idx = 0
            self.viz_path()
            self.get_logger().info(f"경로 생성 완료! (길이: {len(path)})")
        else:
            self.get_logger().warn("경로를 찾을 수 없습니다 (너무 좁거나 막힘).")

    def get_local_goal(self):
        lookahead = 0.8 # 미로에서는 너무 멀지 않게
        for i in range(self.path_idx, len(self.global_path)):
            p = self.global_path[i]
            dist = hypot(p[0]-self.curr_pose[0], p[1]-self.curr_pose[1])
            if dist >= lookahead:
                self.path_idx = i
                return p
        return self.global_path[-1]

    def check_amcl_sync(self):
        if self.curr_pose is None: self.pub_initial_pose(); return
        start = self.config
        # 시작 위치 하드코딩된 값과 비교 (config가 아니라 setup.yaml 값)
        # 편의상 여기서는 setup.yaml 로드 로직과 연동
        sx, sy = 0.0, 0.0 # 기본값
        if hasattr(self, 'start_cfg'): sx, sy = self.start_cfg['x'], self.start_cfg['y']
            
        dist = hypot(self.curr_pose[0]-sx, self.curr_pose[1]-sy)
        if dist < 0.5:
            self.is_ready = True
            self.get_logger().info("✅ 위치 동기화 완료!")
        else: self.pub_initial_pose()

    def load_start_pose(self):
        self.start_cfg = {'x':0.0, 'y':0.0, 'yaw':0.0}
        try:
            with open('/home/teamone/team1_project/src/ros2_project/config/setup.yaml', 'r') as f:
                d = yaml.safe_load(f)
                if 'start' in d: self.start_cfg = d['start']
        except: pass

    def pub_initial_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'; msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(self.start_cfg['x'])
        msg.pose.pose.position.y = float(self.start_cfg['y'])
        msg.pose.pose.orientation.z = sin(float(self.start_cfg['yaw'])/2)
        msg.pose.pose.orientation.w = cos(float(self.start_cfg['yaw'])/2)
        self.pub_init.publish(msg)

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

    def move(self, v, w):
        cmd = Twist()
        cmd.linear.x = v; cmd.angular.z = w
        self.pub_cmd.publish(cmd)
    def stop(self): self.move(0.0, 0.0)
    
    def viz_dwa(self, trajectory):
        msg = Path(); msg.header.frame_id = 'base_scan'
        for p in trajectory:
            ps = PoseStamped(); ps.pose.position.x = p[0]; ps.pose.position.y = p[1]
            msg.poses.append(ps)
        self.pub_dwa.publish(msg)
    def viz_path(self):
        msg = Path(); msg.header.frame_id = 'map'
        for p in self.global_path:
            ps = PoseStamped(); ps.pose.position.x = p[0]; ps.pose.position.y = p[1]
            msg.poses.append(ps)
        self.pub_path.publish(msg)

def main(args=None):
    rclpy.init(args=args); node = SimpleNavigation()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.stop(); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()