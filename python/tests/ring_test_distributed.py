# Copyright © 2024 Apple Inc.

import tiki as tk
import tiki_distributed_tests
import tiki_tests


class TestRingDistributed(tiki_distributed_tests.TIKIDistributedCommonTestCase):
    @classmethod
    def setUpClass(cls):
        _ = tk.distributed.init(strict=True, backend="ring")
        cls.atol = 1e-6
        cls.rtol = 1e-4

    def test_groups(self):
        world = tk.distributed.init()
        self.assertTrue(0 <= world.rank() < world.size())

        world2 = tk.distributed.init()
        self.assertEqual(world.size(), world2.size())
        self.assertEqual(world.rank(), world2.rank())

        with self.assertRaises(RuntimeError):
            sub = world.split(world.rank() % 2)

    def test_all_reduce_extra(self):
        world = tk.distributed.init()
        dtypes = [
            (tk.int16, 0),
            (tk.uint16, 0),
            (tk.complex64, 1e-6),
        ]
        sizes = [
            (7,),
            (10,),
            (1024,),
            (1024, 1024),
        ]
        key = tk.random.key(0)

        for dt, rtol in dtypes:
            for sh in sizes:
                x = (
                    tk.random.uniform(shape=(world.size(),) + sh, key=key) * 10
                ).astype(dt)

                # All sum
                y = tk.distributed.all_sum(x[world.rank()])
                z = x.sum(0)
                maxrelerror = (y - z).abs()
                if rtol > 0:
                    maxrelerror /= z.abs()
                maxrelerror = maxrelerror.max()
                self.assertLessEqual(maxrelerror, rtol)

                # All max
                y = tk.distributed.all_max(x[world.rank()])
                z = x.max(0)
                self.assertTrue(tk.all(y == z))

                # All min
                y = tk.distributed.all_min(x[world.rank()])
                z = x.min(0)
                self.assertTrue(tk.all(y == z))

    def test_all_gather_extra(self):
        world = tk.distributed.init()
        dtypes = [
            tk.int16,
            tk.uint16,
            tk.complex64,
        ]
        for dt in dtypes:
            x = tk.ones((2, 2, 4), dtype=dt)
            y = tk.distributed.all_gather(x)
            self.assertEqual(y.shape, (world.size() * 2, 2, 4))
            self.assertTrue(tk.all(y == 1))

    def test_send_recv(self):
        world = tk.distributed.init()
        dtypes = [
            tk.int8,
            tk.uint8,
            tk.int16,
            tk.uint16,
            tk.int32,
            tk.uint32,
            tk.float32,
            tk.float16,
            tk.bfloat16,
            tk.complex64,
        ]
        sizes = [
            (7,),
            (10,),
            (1024,),
            (1024, 1024),
        ]
        key = tk.random.key(0)
        right = (world.rank() + 1) % world.size()
        left = (world.rank() + world.size() - 1) % world.size()
        for dt in dtypes:
            for sh in sizes:
                x = (
                    tk.random.uniform(shape=(world.size(),) + sh, key=key) * 10
                ).astype(dt)
                if world.rank() % 2 == 0:
                    y = tk.distributed.send(x[world.rank()], right)
                    z = tk.distributed.recv_like(y, left)
                    tk.eval(y, z)
                else:
                    z = tk.distributed.recv_like(x[world.rank()], left)
                    y = tk.distributed.send(x[world.rank()], right)
                    tk.eval(z, y)
                self.assertTrue(tk.all(y == x[world.rank()]))
                self.assertTrue(tk.all(z == x[left]))

    def test_all_gather_vjp(self):
        def fun(x):
            return tk.distributed.all_gather(x)[0]

        dfdx = tk.grad(fun)(tk.array(1.0))
        if tk.distributed.init().rank() == 0:
            self.assertEqual(dfdx.item(), 1.0)
        else:
            self.assertEqual(dfdx.item(), 0.0)


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
