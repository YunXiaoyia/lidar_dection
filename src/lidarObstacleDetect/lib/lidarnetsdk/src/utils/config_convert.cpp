#include <string>

#include "log/logging.h"
#include "ryml_all.hpp"
#include "common/toml.hpp"
#include "common/json.hpp"
#include "log/logger.h"
#include "config_convert.h"

namespace lidar_net {

std::string convert2json(std::string_view config_path, std::string_view config_type) {
    std::string config_type_cpy(config_type);
    std::transform(config_type_cpy.begin(), config_type_cpy.end(), config_type_cpy.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (config_type_cpy == "toml") {
        const auto param_node = toml::parse_file(config_path);
        std::stringstream ss;
        ss << toml::json_formatter{param_node} << "\n";
        return ss.str();
    } else if (config_type_cpy == "yaml") {
        std::string param_node_str;
        std::ifstream file(config_path.data());
        if (file.is_open()) {
            param_node_str = std::string((std::istreambuf_iterator<char>(file)), (std::istreambuf_iterator<char>()));
            file.close();
        } else {
            PLOG_ERROR << "[Config]无法打开文件: " << config_path;
            return "";
        }
        const std::string param_node_str_str(param_node_str);
        const ryml::Tree tree = ryml::parse_in_arena(ryml::to_csubstr(param_node_str_str));
        return ryml::emitrs_json<std::string>(tree);
    } else if (config_type_cpy == "json") {
        const auto param_node = nlohmann::json::parse(config_path);
        return std::string(param_node);
    }

    PLOG_ERROR << "[Config] Unknown config type. Should be json, toml or yaml, but get: " << config_type_cpy;
    return "";
}

std::string config_convert(std::string_view param_node_str, std::string_view config_type) {
    std::string config_type_cpy(config_type);
    std::transform(config_type_cpy.begin(), config_type_cpy.end(), config_type_cpy.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (config_type_cpy == "toml") {
        const auto param_node = toml::parse(param_node_str);
        std::stringstream ss;
        ss << toml::json_formatter{param_node} << "\n";
        return ss.str();
    } else if (config_type_cpy == "yaml") {
        const std::string param_node_str_str(param_node_str);
        const ryml::Tree tree = ryml::parse_in_arena(ryml::to_csubstr(param_node_str_str));
        return ryml::emitrs_json<std::string>(tree);
    } else if (config_type_cpy == "json") {
        return std::string(param_node_str);
    }

    PLOG_ERROR << "[Config] Unknown config type. Should be json, toml or yaml, but get: " << config_type_cpy;
    return std::string(param_node_str);
}

}  // namespace lidar_net