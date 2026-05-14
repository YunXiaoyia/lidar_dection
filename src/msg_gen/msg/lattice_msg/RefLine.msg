#一条完整的从起点到终点的参考线,车道级别结果,只发送一次
std_msgs/Header header  #此条信息发出的时间
int32 refline_id       #此条路径的版本号
LaneSegment[] segment_list  #路段列表，按先后顺序
PathPoint start_point
PathPoint end_point