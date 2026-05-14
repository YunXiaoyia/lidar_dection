#参考线单点级别信息
float64 local_x
float64 local_y
float64 local_yaw    #航向角
float64 frenet_s     #沿着参考线从起点开始的累积弧长
float64 kappa        #曲率
float64 dkappa       #曲率变化率
float64 left_boundary       #左跨越容许距离(若不可跨越，则为车道宽的一半)
float64 right_boundary       #右跨越容许距离


