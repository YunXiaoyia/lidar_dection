bool VCU_LowValtError       #1830E621VCU欠压
bool VCU_HighValtError      #1830E621VCU超限
bool VCU_LowValt            #1830E621VCU供电低
bool VCU_HighValt           #1830E621VCU供电高
bool Engine_LowValt         #1830E621发动机欠压
bool EPS_LowOil             #1830E621EPS流量不足
bool Engine_StartFail       #1830E621发动机启动失败
bool AngleSensor_Offline    #1830E621车架倾角仪掉线
bool LiftSensor_Offline     #1830E621货箱倾角仪掉线
bool ECU_Offline            #1830E621掉线——ECU
bool TCU_Offline            #1830E621掉线——TCU
bool EBS_Offline            #1830E621掉线——EBS
bool EPS_Offline            #1830E621掉线——EPS
bool EPB_Offline            #1830E621掉线——EPB
bool ADCU_HearbeatError     #1830E621ADCU心跳异常
bool ADCU_Lost              #1830E621ADCU掉线
bool GearError_A2M          #1830E621一键切换时，变速箱未切换成功
bool AirJar_LowPressure
bool A2M_Switch_Error
bool SelfDrivingCANError
bool CAN1Error
bool CAN2Error
int8 flame_out_error        #1830E621熄火故障
int8 Engine_StartFail_count #1830E621连续启动失败
int8 Failure_Class          #18FECA21整车故障等级
int8 VCU_error              #1830C021VCU故障
int8 Switch_Error
int8 Engine_Error           #1830E021发动机故障
int8 Gearbox_Error          #1830E021变速箱故障
int8 EBS_Error              #1830E021EBS故障
int8 EPS_Error              #1830E021EPS故障
int8 EPB_Error              #1830E021EPB故障
int8 Electric_Error         #1830E021电气系统故障等级
int8 Tire_Pressure_Error    #1830E021胎压故障等级
int16 Chassis_Error
int8 Flame_OUT
int8 CAN_Error
int8 Posture_Angle_Error
string ISG_Error_Code
string Cluth_Error_Code
string Main_Motor_Error_Code
string BMS_1_Error_Code
string controller_Error_Code
time HeartBeat
int16 CarFalut_Class             #18FECA21 整车故障等级
int16 Engine_ErrorCode         #18FECA21发动机故障码
int16 GearBox_ErrorCode        #18FECA21变速箱故障码
int16 Break_ErrorCode          #18FECB21刹车故障码
int8 ElectricSYS_ErrorCode    #18FECB21电气系统故障码
int8 EPB_ErrorCode            #18FECB21EPB故障码
int16 EPS_ErrorCode            #18FECB21EPS故障码
