#lattice的状态控制，决策层应按固定频率持续下发至lattice
bool enable_lattice     #true代表启动，false代表停用并实施急停
float64 max_deacc       #停用时，所允许的最大减速度
float64 max_acc         #启用时，所允许的最大加速度(起步安全限制)