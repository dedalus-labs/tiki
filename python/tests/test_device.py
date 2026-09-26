# Copyright © 2023 Apple Inc.

import unittest

import tiki as tk
import tiki_tests


# Don't inherit from TIKITestCase to avoid call to setUp
class TestDefaultDevice(unittest.TestCase):
    def test_tiki_default_device(self):
        device = tk.default_device()
        if tk.is_available(tk.gpu):
            self.assertEqual(device, tk.Device(tk.gpu))
            self.assertEqual(str(device), "Device(gpu, 0)")
            self.assertEqual(device, tk.gpu)
            self.assertEqual(tk.gpu, device)
        else:
            self.assertEqual(device.type, tk.Device(tk.cpu))
            with self.assertRaises(ValueError):
                tk.set_default_device(tk.gpu)


class TestDevice(tiki_tests.TIKITestCase):
    def test_device(self):
        device = tk.default_device()

        cpu = tk.Device(tk.cpu)
        tk.set_default_device(cpu)
        self.assertEqual(tk.default_device(), cpu)
        self.assertEqual(str(cpu), "Device(cpu, 0)")

        tk.set_default_device(tk.cpu)
        self.assertEqual(tk.default_device(), tk.cpu)
        self.assertEqual(cpu, tk.cpu)
        self.assertEqual(tk.cpu, cpu)

        # Restore device
        tk.set_default_device(device)

    @unittest.skipIf(not tk.is_available(tk.gpu), "GPU is not available")
    def test_device_context(self):
        default = tk.default_device()
        diff = tk.cpu if default == tk.gpu else tk.gpu
        self.assertNotEqual(default, diff)
        with tk.stream(diff):
            a = tk.add(tk.zeros((2, 2)), tk.ones((2, 2)))
            tk.eval(a)
            self.assertEqual(tk.default_device(), diff)
        self.assertEqual(tk.default_device(), default)

    def test_op_on_device(self):
        x = tk.array(1.0)
        y = tk.array(1.0)

        a = tk.add(x, y, stream=None)
        b = tk.add(x, y, stream=tk.default_device())
        self.assertEqual(a.item(), b.item())
        b = tk.add(x, y, stream=tk.cpu)
        self.assertEqual(a.item(), b.item())

        if tk.metal.is_available():
            b = tk.add(x, y, stream=tk.gpu)
            self.assertEqual(a.item(), b.item())


class TestStream(tiki_tests.TIKITestCase):
    def test_stream(self):
        s1 = tk.default_stream(tk.default_device())
        self.assertEqual(s1.device, tk.default_device())

        s2 = tk.new_stream(tk.default_device())
        self.assertEqual(s2.device, tk.default_device())
        self.assertNotEqual(s1, s2)

        if tk.is_available(tk.gpu):
            s_gpu = tk.default_stream(tk.gpu)
            self.assertEqual(s_gpu.device, tk.gpu)
        else:
            with self.assertRaises(ValueError):
                tk.default_stream(tk.gpu)

        s_cpu = tk.default_stream(tk.cpu)
        self.assertEqual(s_cpu.device, tk.cpu)

        s_cpu = tk.new_stream(tk.cpu)
        self.assertEqual(s_cpu.device, tk.cpu)

        if tk.is_available(tk.gpu):
            s_gpu = tk.new_stream(tk.gpu)
            self.assertEqual(s_gpu.device, tk.gpu)
        else:
            with self.assertRaises(ValueError):
                tk.new_stream(tk.gpu)

    def test_op_on_stream(self):
        x = tk.array(1.0)
        y = tk.array(1.0)

        a = tk.add(x, y, stream=tk.default_stream(tk.default_device()))

        if tk.is_available(tk.gpu):
            b = tk.add(x, y, stream=tk.default_stream(tk.gpu))
            self.assertEqual(a.item(), b.item())
            s_gpu = tk.new_stream(tk.gpu)
            b = tk.add(x, y, stream=s_gpu)
            self.assertEqual(a.item(), b.item())

        b = tk.add(x, y, stream=tk.default_stream(tk.cpu))
        self.assertEqual(a.item(), b.item())
        s_cpu = tk.new_stream(tk.cpu)
        b = tk.add(x, y, stream=s_cpu)
        self.assertEqual(a.item(), b.item())


class TestDeviceInfo(tiki_tests.TIKITestCase):
    def test_device_count(self):
        cpu_count = tk.device_count(tk.cpu)
        self.assertIsInstance(cpu_count, int)
        self.assertEqual(cpu_count, 1)

        gpu_count = tk.device_count(tk.gpu)
        self.assertIsInstance(gpu_count, int)
        self.assertGreaterEqual(gpu_count, 0)

    def test_device_info_cpu(self):
        info = tk.device_info(tk.cpu)
        self.assertIsInstance(info, dict)
        self.assertIn("device_name", info)
        self.assertTrue(len(info["device_name"]) > 0)
        self.assertIn("architecture", info)

    @unittest.skipIf(not tk.is_available(tk.gpu), "GPU is not available")
    def test_device_info_gpu(self):
        gpu_count = tk.device_count(tk.gpu)
        for i in range(gpu_count):
            info = tk.device_info(tk.Device(tk.gpu, i))
            self.assertIsInstance(info, dict)
            self.assertIn("device_name", info)
            self.assertTrue(len(info["device_name"]) > 0)
            self.assertIn("architecture", info)

    def test_device_info_default(self):
        info = tk.device_info()
        self.assertIsInstance(info, dict)
        self.assertIn("device_name", info)


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
