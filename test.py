import dexcontrol

robot = dexcontrol.Robot()

left_qpos = robot.left_arm.get_joint_pos()
right_qpos = robot.right_arm.get_joint_pos()
torso_qpos = robot.torso.get_joint_pos()
head_qpos = robot.head.get_joint_pos()
