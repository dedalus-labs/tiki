// Copyright © 2023-2025 Apple Inc.

#include <optional>
#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>

#include "tiki/device.h"
#include "tiki/utils.h"

namespace tk = tiki::core;
namespace nb = nanobind;
using namespace nb::literals;

void init_device(nb::module_& m) {
  auto device_class = nb::class_<tk::Device>(
      m, "Device", R"pbdoc(A device to run operations on.)pbdoc");
  nb::enum_<tk::Device::DeviceType>(m, "DeviceType")
      .value("cpu", tk::Device::DeviceType::cpu)
      .value("gpu", tk::Device::DeviceType::gpu)
      .export_values()
      .def(
          "__eq__",
          [](const tk::Device::DeviceType& d, const nb::object& other) {
            if (!nb::isinstance<tk::Device>(other) &&
                !nb::isinstance<tk::Device::DeviceType>(other)) {
              return false;
            }
            return d == nb::cast<tk::Device>(other);
          });

  device_class
      .def(nb::init<tk::Device::DeviceType, int>(), "type"_a, "index"_a = 0)
      .def_ro("type", &tk::Device::type)
      .def(
          "__repr__",
          [](const tk::Device& d) {
            std::ostringstream os;
            os << d;
            return os.str();
          })
      .def("__eq__", [](const tk::Device& d, const nb::object& other) {
        if (!nb::isinstance<tk::Device>(other) &&
            !nb::isinstance<tk::Device::DeviceType>(other)) {
          return false;
        }
        return d == nb::cast<tk::Device>(other);
      });

  nb::implicitly_convertible<tk::Device::DeviceType, tk::Device>();

  m.def(
      "default_device",
      &tk::default_device,
      R"pbdoc(Get the default device.)pbdoc");
  m.def(
      "set_default_device",
      &tk::set_default_device,
      "device"_a,
      nb::sig("def set_default_device(device: Device | DeviceType) -> None"),
      R"pbdoc(Set the default device.)pbdoc");
  m.def(
      "is_available",
      &tk::is_available,
      "device"_a,
      nb::sig("def is_available(device: Device | DeviceType) -> bool"),
      R"pbdoc(Check if a back-end is available for the given device.)pbdoc");
  m.def(
      "device_count",
      &tk::device_count,
      "device_type"_a,
      R"pbdoc(
      Get the number of available devices for the given device type.

      Args:
          device_type (DeviceType): The type of device to query (cpu or gpu).

      Returns:
          int: Number of devices.
      )pbdoc");
  m.def(
      "device_info",
      [](std::optional<tk::Device> d) {
        return tk::device_info(d.value_or(tk::default_device()));
      },
      "d"_a = nb::none(),
      nb::sig(
          "def device_info(d: None | Device | DeviceType = None) -> dict[str, str | int]"),
      R"pbdoc(
      Get information about a device.

      Returns a dictionary with device properties. Available keys depend
      on the backend and device type. Common keys include ``device_name``,
      ``architecture``, and ``total_memory`` (or ``memory_size``).

      Args:
          d (Device): The device to query (defaults to the default device).

      Returns:
          dict: Device information.
      )pbdoc");
}
