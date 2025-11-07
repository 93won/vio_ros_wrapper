from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('vio_ros_wrapper')
    rviz_config_file = os.path.join(pkg_share, 'rviz', 'rgbd.rviz')
    default_config_file = os.path.join(pkg_share, 'config', 'd435i_cafe.yaml')
    
    return LaunchDescription([
        # Declare launch arguments for RGBD Visual Odometry (VO)
        # Note: RGBD-VO mode does not use IMU data
        DeclareLaunchArgument(
            'rgb_image_topic',
            default_value='/camera/rgb/image_raw',
            description='RGB camera image topic'
        ),
        DeclareLaunchArgument(
            'depth_image_topic',
            default_value='/camera/depth/image_raw',
            description='Depth camera image topic'
        ),
        DeclareLaunchArgument(
            'queue_size',
            default_value='10000',
            description='Subscriber queue size'
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_file,
            description='VIO configuration file path (defaults to d435i_cafe.yaml)'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RViz2'
        ),

        # RGBD-VO Node (Visual Odometry using RGB+Depth, no IMU)
        Node(
            package='vio_ros_wrapper',
            executable='vio_node',
            name='vio_node',
            output='screen',
            parameters=[{
                'rgb_image_topic': LaunchConfiguration('rgb_image_topic'),
                'depth_image_topic': LaunchConfiguration('depth_image_topic'),
                'queue_size': LaunchConfiguration('queue_size'),
                'config_file': LaunchConfiguration('config_file'),
            }]
        ),
        
        # RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file],
            condition=IfCondition(LaunchConfiguration('use_rviz'))
        ),
    ])
