// Copyright © 2023-2024 Apple Inc.

#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "python/src/random.h"
#include "tiki/stream.h"
#include "tiki/utils.h"

namespace tk = tiki::core;
namespace nb = nanobind;
using namespace nb::literals;

// Create the StreamContext on enter and delete on exit.
class PyStreamContext {
 public:
  PyStreamContext(tk::StreamOrDevice s) : _inner(nullptr) {
    if (std::holds_alternative<std::monostate>(s)) {
      throw std::runtime_error(
          "[StreamContext] Invalid argument, please specify a stream or device.");
    }
    _s = s;
  }

  void enter() {
    _inner = new tk::StreamContext(_s);
  }

  void exit() {
    if (_inner != nullptr) {
      delete _inner;
      _inner = nullptr;
    }
  }

 private:
  tk::StreamOrDevice _s;
  tk::StreamContext* _inner;
};

void init_stream(nb::module_& m) {
  nb::class_<tk::Stream>(
      m,
      "Stream",
      R"pbdoc(
      A stream for running operations on a given device.
      )pbdoc")
      .def_ro("device", &tk::Stream::device)
      .def(
          "__repr__",
          [](const tk::Stream& s) {
            std::ostringstream os;
            os << s;
            return os.str();
          })
      .def("__eq__", [](const tk::Stream& s, const nb::object& other) {
        return nb::isinstance<tk::Stream>(other) &&
            s == nb::cast<tk::Stream>(other);
      });

  nb::class_<tk::ThreadLocalStream>(
      m,
      "ThreadLocalStream",
      R"pbdoc(
      A stream that will be unique per thread and can be used to run operations on a given device.
      )pbdoc")
      .def_ro("device", &tk::ThreadLocalStream::device)
      .def(
          "__repr__",
          [](const tk::ThreadLocalStream& s) {
            std::ostringstream os;
            os << "ThreadLocalStream(" << s.device << ", " << s.index << ")";
            return os.str();
          })
      .def(
          "__eq__",
          [](const tk::ThreadLocalStream& s, const nb::object& other) {
            return nb::isinstance<tk::ThreadLocalStream>(other) &&
                s == nb::cast<tk::ThreadLocalStream>(other);
          });

  nb::implicitly_convertible<tk::Device::DeviceType, tk::Device>();

  m.def(
      "default_stream",
      &tk::default_stream,
      "device"_a,
      nb::sig("def default_stream(device: Device | DeviceType) -> Stream"),
      R"pbdoc(Get the device's default stream.)pbdoc");
  m.def(
      "set_default_stream",
      &tk::set_default_stream,
      "stream"_a,
      R"pbdoc(
        Set the default stream.

        This will make the given stream the default for the
        streams device. It will not change the default device.

        Args:
          stream (stream): Stream to make the default.
      )pbdoc");
  m.def(
      "new_stream",
      &tk::new_stream,
      "device"_a,
      nb::sig("def new_stream(device: Device | DeviceType) -> Stream"),
      R"pbdoc(
        Make a new stream on the given device.

        The stream can only be used on the thread where it was created on, using
        it in any other thread would result in errors.
      )pbdoc");
  m.def(
      "new_thread_unsafe_stream",
      &tk::new_thread_unsafe_stream,
      "device"_a,
      nb::sig(
          "def new_thread_unsafe_stream(device: Device | DeviceType) -> Stream"),
      R"pbdoc(
        Make a new stream that can be used in any thread.

        Unlike :func:`new_stream` which can only work on the thread of creation,
        streams created by this API can be passed to and evaluated anywhere, but
        note that currently all nodes in a graph must be evaluated in sequence
        and it is user's responsibilty to ensure there is no race condition.
      )pbdoc");
  m.def(
      "new_thread_local_stream",
      &tk::new_thread_local_stream,
      "device"_a,
      nb::sig(
          "def new_thread_local_stream(device: Device | DeviceType) -> ThreadLocalStream"),
      R"pbdoc(Make a new stream that will be unique per thread.)pbdoc");
  m.def(
      "clear_streams",
      []() {
        reset_random_state();
        nb::gil_scoped_release nogil;
        tk::clear_streams();
      },
      R"pbdoc(Destroy all streams created in current thread.)pbdoc");

  nb::class_<PyStreamContext>(m, "StreamContext", R"pbdoc(
        A context manager for setting the current device and stream.

        See :func:`stream` for usage.

        Args:
            s: The stream or device to set as the default.
  )pbdoc")
      .def(nb::init<tk::StreamOrDevice>(), "s"_a)
      .def("__enter__", [](PyStreamContext& scm) { scm.enter(); })
      .def(
          "__exit__",
          [](PyStreamContext& scm,
             const std::optional<nb::type_object>& exc_type,
             const std::optional<nb::object>& exc_value,
             const std::optional<nb::object>& traceback) { scm.exit(); },
          "exc_type"_a = nb::none(),
          "exc_value"_a = nb::none(),
          "traceback"_a = nb::none());
  m.def(
      "stream",
      [](tk::StreamOrDevice s) { return PyStreamContext(s); },
      "s"_a,
      R"pbdoc(
        Create a context manager to set the default device and stream.

        Args:
            s: The :obj:`Stream` or :obj:`Device` to set as the default.

        Returns:
            A context manager that sets the default device and stream.

        Example:

        .. code-block::python

          import tiki as tk

          # Create a context manager for the default device and stream.
          with tk.stream(tk.cpu):
              # Operations here will use tk.cpu by default.
              pass
      )pbdoc");
  m.def(
      "synchronize",
      [](tk::StreamOrDevice s) {
        nb::gil_scoped_release nogil;
        if (std::holds_alternative<std::monostate>(s)) {
          tk::synchronize();
        } else {
          tk::synchronize(tk::to_stream(s));
        }
      },
      "stream"_a = nb::none(),
      R"pbdoc(
      Synchronize with the given stream.

      Args:
        stream (Stream, optional): Stream to synchronize. If device is
           provided the default stream for that device is used. If ``None``
           then the default stream of the default device is used.
           Default: ``None``.
      )pbdoc");
}
