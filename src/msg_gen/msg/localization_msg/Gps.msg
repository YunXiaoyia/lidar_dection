std_msgs/Header header
int32 gps_time
bool state
float64 lat   # 前轴定位经度
float64 lon   # 前轴定位纬度
float64 posX  # 前轴定位信息X
float64 posY  # 前轴定位信息Y
float64 posZ  # 前轴定位信息Z
float64 Rear_posX # 后轴定位信息X
float64 Rear_posY # 后轴定位信息Y
float64 Rear_posZ # 后轴定位信息Z
float64 oriX    #roll
float64 oriY    #pitch
float64 oriZ    #yaw 以东向为0度，北向90度，南向-90度，西向是180/-180度
float64 velX
float64 velY
float64 velZ
float64 speed  # 车辆速度 km/h
float64 accelX
float64 accelY
float64 accelZ
float64 angVelX
float64 angVelY
float64 angVelZ