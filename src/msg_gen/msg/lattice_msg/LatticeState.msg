#lattice的状态反馈，按固定频率上报至决策层

bool running            #运行中标志
int32 state_machine     #运行状态(0-无障碍寻迹，1-有障碍跟车，2-静态障碍正在绕行，3-终点停车，4-有障碍无法绕行)
int32 fault_info        #错误状态(0-无错误，1-自定义错误分支)
int32 exec_refline_id      #正在寻迹的路径版本号
