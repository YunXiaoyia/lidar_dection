add_executable(dsvt_eval 
  dsvt_eval.cpp
  json_operate.cpp)
target_link_libraries(dsvt_eval 
  lidar-net-sdk
  stdc++fs)
# 忽略dsvt_eval编译时产生的warnings
target_compile_options(dsvt_eval PRIVATE -w)