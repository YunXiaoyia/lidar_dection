std_msgs/Header header

float64 AngleHeading_slam  #-180到180
float64 AnglePitch_slam
float64 AngleRoll_slam

float64 PosLon_slam
float64 PosLat_slam
float64 PosAlt_slam

float64 slam_reliability  #匹配可靠性
float64 slam_exe_time  #匹配耗时
float64 slam_fitness_score  #匹配得分
float64 slam_has_converged #正态分布收敛值
int32   slam_iteration #迭代次数
int32   slam_use_predict_pose

#odom
float64 localpose_x			# 高斯投影位置,m
float64 localpose_y
float64 localpose_z
float64 localpose_pitch		# 俯仰角,x轴方向(正右方)右手定则四指方向为正,单位为0.01度
float64 localpose_roll		# 翻滚角,y轴方向(正前方)右手定则四指方向为正,单位为0.01度
float64 localpose_azimuth	# 航向角，单位为0.01度,向东为零度，逆时针0-360
int32 northVelocity			# north速度，单位为cm/s
int32 eastVelocity			# east速度,单位为cm/s
int32 upVelocity			# up速度,单位为cm/s
float32 speed 				# m/s
float32 roll_rate           # x轴角速度rad/s
float32 pitch_rate          # y轴角速度rad/s
float32 yaw_rate           	# z轴角速度rad/s
float32 acc_x               # x轴加速度m^2/s
float32 acc_y               # y轴加速度m^2/s
float32 acc_z               # z轴加速度m^2/s

