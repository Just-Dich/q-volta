// CPU engine for Inkling-2-Mini (arch "inkling2_loop"): extra introspection interface (tests / benchmarks).
// Obtain with dynamic_cast<qv::InklingCpuInfo*>(engine.get()).
#pragma once
#include <cstdint>
#include <vector>

namespace qv {

struct InklingCpuInfo {
    virtual ~InklingCpuInfo() = default;
    // weight bytes read by one decode token as the kernels see them (planes + expanded scale tables + fp16 side matrices)
    virtual double decode_bytes_per_token() const = 0;
    // router decisions of the last forward call (EngineOptions::dump_hidden): int8 [48 applications][T][8]
    virtual const std::vector<int8_t>& router_dump() const = 0;
    // per-phase wall time accumulated since the last call (ms), only when QV_INK_PROF=1; names and values in `names`/`ms`
    virtual void phase_report(std::vector<const char*>& names, std::vector<double>& ms, std::vector<long>& calls) = 0;
};

}  // namespace qv
