// INI model presets (--models-preset PATH) for on-demand multi-model routing in qv_server.
#pragma once
#include <string>
#include <vector>

#include "server/options.h"

namespace qv::server {

struct ModelPreset {
    std::string id;
    Options opts;
};

// Parse `path` (llama-server presets.ini format with `[*]` global defaults and `[model-id]` sections).
// Throws std::runtime_error on I/O or syntax errors, or if a named preset has no `model` path.
std::vector<ModelPreset> load_presets_ini(const std::string& path, const Options& base, std::vector<std::string>& warnings);

}  // namespace qv::server
