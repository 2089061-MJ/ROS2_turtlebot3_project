import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 패키지 경로 및 설정 가져오기
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    # map_server가 로드할 맵 파일 경로 
    default_map_path = '/home/teamone/map.yaml'

    # 2. Launch Argument 설정
    # use_sim_time: 실제 터틀봇 환경이면 'false', 가제보 시뮬레이션이면 'true'
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    map_yaml_file = LaunchConfiguration('map', default=default_map_path)
    params_file = LaunchConfiguration('params_file', default=os.path.join(nav2_bringup_dir, 'params', 'nav2_params.yaml'))

    # 3. Localization (Map Server + AMCL) 실행
    # Nav2의 표준 런치 파일을 가져와서 실행합니다.
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'localization_launch.py')
        ),
        launch_arguments={
            'map': map_yaml_file,
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'autostart': 'true',  # [중요] 맵 서버와 AMCL을 자동으로 Active 상태로 만듦
            'use_lifecycle_mgr': 'false' 
        }.items()
    )

    # 4. RViz2 실행
    rviz_config_dir = os.path.join(nav2_bringup_dir, 'rviz', 'nav2_default_view.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # 5. [사용자 정의] 통합 내비게이션 노드 실행
    # setup.py의 entry_points에 'nav = ...'로 등록되어 있으므로 executable='nav' 입니다.
    integrated_nav_node = Node(
        package='ros2_project',      # package.xml의 패키지 이름
        executable='nav2',            # setup.py의 entry_points 이름
        name='integrated_navigation_apf',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 6. 실행 목록 반환
    return LaunchDescription([
        # Arguments 선언
        DeclareLaunchArgument(
            'map',
            default_value=default_map_path,
            description='Full path to map yaml file to load'),
        
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(nav2_bringup_dir, 'params', 'nav2_params.yaml'),
            description='Full path to the ROS2 parameters file to use'),

        # 실행할 노드들
        localization_launch,
        rviz_node,
        integrated_nav_node
    ])