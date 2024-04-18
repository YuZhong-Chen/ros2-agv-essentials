from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

ARGUMENTS = [
    DeclareLaunchArgument(
        "launch_rviz",
        default_value="False",
        description="Launch rviz2, by default is False",
    ),
    DeclareLaunchArgument(
        "spawn_kobuki",
        default_value="True",
        description="Spawn kobuki, by default is True",
    ),
]

def generate_launch_description():
    # Get gazebo world file path
    world_file = PathJoinSubstitution(
        [
            FindPackageShare("kobuki_gazebo"),
            "worlds/aws_hospital",
            "hospital.world",
        ],
    )

    # Launch Gazebo
    launch_gazebo = IncludeLaunchDescription(
        PathJoinSubstitution(
            [
                FindPackageShare("kobuki_gazebo"),
                "launch",
                "gazebo.launch.py",
            ],
        ),
        launch_arguments={
            "world_path": world_file,
            "launch_rviz": LaunchConfiguration("launch_rviz"),
            "spawn_kobuki": LaunchConfiguration("spawn_kobuki"),
            "robot_init_x": "0.0",
            "robot_init_y": "1.0",
            "robot_init_z": "0.0",
            "robot_init_yaw": "1.57",
        }.items(),
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(launch_gazebo)

    return ld
