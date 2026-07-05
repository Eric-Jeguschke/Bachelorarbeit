from launch import LaunchDescription
import launch_ros.actions


def generate_launch_description():
    """Startet alle Nodes des Systems (db, microphone, speaker, recognition, voice_ai)."""
    return LaunchDescription([
        launch_ros.actions.Node(package="db", executable="db_node", output="screen"),
        launch_ros.actions.Node(package="microphone", executable="microphone_node", output="screen"),
        launch_ros.actions.Node(package="speaker", executable="speaker_node", output="screen"),
        launch_ros.actions.Node(package="recognition", executable="recognition_node", output="screen"),
        launch_ros.actions.Node(package="voice_ai", executable="voice_node", output="screen"),
    ])
