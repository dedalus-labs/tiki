"""The native export signature must agree with the specialization profile."""

import unittest

from mlx import core

from tiki_compiler.capture import from_events, parse_primitive, trace
from tiki_compiler.events import PrimitiveEvent
from tiki_compiler.graph import Profile, UnsupportedGraphError


def identity(value: core.array) -> core.array:
    return value


class CaptureContractTests(unittest.TestCase):
    def test_profile_shape_matches_exported_input(self) -> None:
        events = trace(identity, (Profile(shape=(4,), strides=(1,)),))
        with self.assertRaisesRegex(UnsupportedGraphError, "profile.*shape"):
            from_events(events, (Profile(shape=(5,), strides=(1,)),))

    def test_signature_headers_occur_exactly_once(self) -> None:
        events = trace(identity, (Profile(shape=(4,), strides=(1,)),))
        header = next(event for event in events if event["type"] == "inputs")
        with self.assertRaisesRegex(UnsupportedGraphError, "duplicate.*inputs"):
            from_events([*events, header], (Profile(shape=(4,), strides=(1,)),))

    def test_collective_requires_its_communicator(self) -> None:
        event: PrimitiveEvent = {
            "type": "primitive",
            "name": "AllReduce",
            "arguments": [2],
            "inputs": [("input", (4,), core.float32)],
            "outputs": [("output", (4,), core.float32)],
            "stream": core.default_stream(core.default_device()),
        }
        with self.assertRaisesRegex(UnsupportedGraphError, "missing communicator"):
            parse_primitive(event)

    def test_stateless_arithmetic_rejects_unmodeled_arguments(self) -> None:
        event: PrimitiveEvent = {
            "type": "primitive",
            "name": "Add",
            "arguments": [1],
            "inputs": [("left", (4,), core.float32), ("right", (4,), core.float32)],
            "outputs": [("output", (4,), core.float32)],
            "stream": core.default_stream(core.default_device()),
        }
        with self.assertRaisesRegex(UnsupportedGraphError, "Add.*arguments"):
            parse_primitive(event)


if __name__ == "__main__":
    unittest.main()
