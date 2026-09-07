// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <unordered_map>
#include <variant>

#include "tiki/api.h"
#include "tiki/array.h"
#include "tiki/io/load.h"
#include "tiki/stream.h"
#include "tiki/utils.h"

namespace tiki::core {
using GGUFMetaData =
    std::variant<std::monostate, array, std::string, std::vector<std::string>>;
using GGUFLoad = std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, GGUFMetaData>>;
using SafetensorsLoad = std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string>>;

/** Save array to out stream in .npy format */
TIKI_API void save(std::shared_ptr<io::Writer> out_stream, array a);

/** Save array to file in .npy format */
TIKI_API void save(std::string file, array a);

/** Load array from reader in .npy format */
TIKI_API array
load(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s = {});

/** Load array from file in .npy format */
TIKI_API array load(std::string file, StreamOrDevice s = {});

/** Load array map from .safetensors file format */
TIKI_API SafetensorsLoad
load_safetensors(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s = {});
TIKI_API SafetensorsLoad
load_safetensors(const std::string& file, StreamOrDevice s = {});

TIKI_API void save_safetensors(
    std::shared_ptr<io::Writer> in_stream,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string> metadata = {});
TIKI_API void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string> metadata = {});

/** Load array map and metadata from .gguf file format */

TIKI_API GGUFLoad load_gguf(const std::string& file, StreamOrDevice s = {});

TIKI_API void save_gguf(
    std::string file,
    std::unordered_map<std::string, array> array_map,
    std::unordered_map<std::string, GGUFMetaData> meta_data = {});

} // namespace tiki::core
