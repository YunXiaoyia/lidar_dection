#分段车道级别信息,以某一段的某一个车道为最小子集

int32 parent_road          #所在road的id
int32 predecessor_road      # 所在road前驱者的id
int32 successor_road      # 所在road后继者的id

int32 segment_id            #分段编号(0.1.2.3.前，中，后，以0为某个road起点的第一段)
int32 lane_id              #车道编号(从左往右，0.1.2.3.4 ，0代表最左侧车道)
PathPoint[] center_line  #代表中心线
int32 speed_limit         #限速
int32 crossable           #是否可跨越(0-双侧不可跨,1-仅左侧可，2仅右侧可，3-双侧可)
