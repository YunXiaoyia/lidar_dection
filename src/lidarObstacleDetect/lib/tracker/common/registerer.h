#pragma once

#include <map>
#include <string>
#include <functional>
#include <memory>

#include "Macros.h"

HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN

// 简单的注册器实现，替代缺失的registerer.h
template<typename Base>
class Registerer {
public:
    using Creator = std::function<std::shared_ptr<Base>()>;
    
    static Registerer& Instance() {
        static Registerer instance;
        return instance;
    }
    
    void Register(const std::string& name, Creator creator) {
        creators_[name] = creator;
    }
    
    std::shared_ptr<Base> Create(const std::string& name) {
        auto it = creators_.find(name);
        if (it != creators_.end()) {
            return it->second();
        }
        return nullptr;
    }
    
    // 添加 GetInstanceByName 方法以兼容现有代码
    static std::shared_ptr<Base> GetInstanceByName(const std::string& name) {
        return Instance().Create(name);
    }
    
private:
    std::map<std::string, Creator> creators_;
};

// 注册宏
// 直接移除这个宏，因为我们不需要显式实例化
#define PERCEPTION_REGISTER_REGISTERER(BaseClass) \
    /* Do nothing - explicit instantiation not needed */

#define PERCEPTION_REGISTER_CLASS(BaseClass, DerivedClass) \
    namespace { \
        struct DerivedClass##_Registrar { \
            DerivedClass##_Registrar() { \
                highway::perception::Registerer<BaseClass>::Instance().Register(#DerivedClass, \
                    []() -> std::shared_ptr<BaseClass> { \
                        return std::make_shared<DerivedClass>(); \
                    }); \
            } \
        }; \
        static DerivedClass##_Registrar g_##DerivedClass##_registrar; \
    }

// 类型定义宏
#define BaseTrackFilterRegisterer highway::perception::Registerer<highway::perception::track::BaseTrackFilter>
#define BaseMotionFilterRegisterer highway::perception::Registerer<highway::perception::track::BaseMotionFilter>
#define BaseShapeFilterRegisterer highway::perception::Registerer<highway::perception::track::BaseShapeFilter>
#define IKalmanFilterRegisterer highway::perception::Registerer<highway::perception::track::IKalmanFilter>

HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
