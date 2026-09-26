// Copyright © 2024 Apple Inc.

#pragma once

#include <optional>

#include "tiki/api.h"
#include "tiki/distributed/distributed.h"
#include "tiki/utils.h"

namespace tiki::core::distributed {

TIKI_API array all_sum(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

TIKI_API array all_gather(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice S = {});

TIKI_API array send(
    const array& x,
    int dst,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

TIKI_API array recv(
    Shape shape,
    Dtype dtype,
    int src,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

TIKI_API array recv_like(
    const array& x,
    int src,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

TIKI_API array all_max(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

TIKI_API array all_min(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

TIKI_API array sum_scatter(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});

} // namespace tiki::core::distributed
