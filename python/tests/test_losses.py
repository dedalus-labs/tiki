# Copyright © 2023 Apple Inc.

import unittest

import numpy as np
import tiki as tk
import tiki.nn as nn
import tiki_tests


class TestLosses(tiki_tests.TIKITestCase):
    def test_cross_entropy(self):
        # No weights, no label smoothing
        logits = tk.array([[0.0, -float("inf")], [-float("inf"), 0.0]])
        indices = tk.array([0, 1])
        expected = tk.array([0.0, 0.0])
        loss = nn.losses.cross_entropy(logits, indices, reduction="none")
        self.assertTrue(tk.allclose(loss, expected))

        # The loss only depends on the gaps between logits, so a large shared
        # offset must not change it
        indices = tk.array([0])
        base = tk.array([[2.0, -1.0]])
        expected = nn.losses.cross_entropy(base, indices, reduction="none")
        for offset in [1e4, 1e6]:
            loss = nn.losses.cross_entropy(base + offset, indices, reduction="none")
            self.assertTrue(tk.allclose(loss, expected, atol=1e-5))

        # Equal logits give log(n) whatever their magnitude
        for v in [1e0, 1e4, 1e8, 1e20]:
            loss = nn.losses.cross_entropy(
                tk.array([[v, v]]), indices, reduction="none"
            )
            self.assertTrue(tk.allclose(loss, tk.array([0.6931472]), atol=1e-5))

        probs = tk.array([[1.0, 0.0], [0.0, 1.0]])
        loss = nn.losses.cross_entropy(logits, probs, reduction="none")
        self.assertTrue(tk.isnan(loss).all())  # produce NaNs, like PyTorch

        # With weights, no label smoothing
        logits = tk.array([[2.0, -1.0], [-1.0, 2.0]])
        indices = tk.array([0, 1])
        weights = tk.array([1.0, 2.0])
        expected = tk.array([0.04858735, 0.0971747])
        loss = nn.losses.cross_entropy(
            logits, indices, weights=weights, reduction="none"
        )
        self.assertTrue(tk.allclose(loss, expected))

        probs = tk.array([[1.0, 0.0], [0.0, 1.0]])
        loss = nn.losses.cross_entropy(logits, probs, weights=weights, reduction="none")
        self.assertTrue(tk.allclose(loss, expected))

        # No weights, with label smoothing
        logits = tk.array([[2.0, -1.0], [-1.0, 2.0]])
        indices = tk.array([0, 1])
        expected = tk.array([0.498587, 0.498587])
        loss = nn.losses.cross_entropy(
            logits, indices, label_smoothing=0.3, reduction="none"
        )
        self.assertTrue(tk.allclose(loss, expected))

        probs = tk.array([[1.0, 0.0], [0.0, 1.0]])
        loss = nn.losses.cross_entropy(
            logits, probs, label_smoothing=0.3, reduction="none"
        )
        self.assertTrue(tk.allclose(loss, expected))

        # With weights and label smoothing
        logits = tk.array([[2.0, -1.0], [-1.0, 2.0]])
        indices = tk.array([0, 1])
        weights = tk.array([1.0, 2.0])
        expected = tk.array([0.49858734, 0.9971747])
        loss = nn.losses.cross_entropy(
            logits, indices, weights=weights, label_smoothing=0.3, reduction="none"
        )
        self.assertTrue(tk.allclose(loss, expected))

        # Test a different axis
        logits = tk.random.normal((4, 8))
        targets = tk.array([1, 2, 3, 0])
        loss = nn.losses.cross_entropy(
            logits.T,
            targets,
            axis=0,
        )
        targets = tk.array([1, 2, 3, 0])
        expected = nn.losses.cross_entropy(
            logits,
            targets,
            axis=-1,
        )
        self.assertTrue(tk.allclose(loss, expected))

    def test_binary_cross_entropy(self):
        def _test_logits_as_inputs():
            logits = tk.array([0.105361, 0.223144, 1.20397, 0.916291])
            targets = tk.array([0, 0, 1, 1])

            # Test with reduction 'none'
            losses_none = nn.losses.binary_cross_entropy(
                logits, targets, reduction="none"
            )
            expected_none = tk.array([0.747215, 0.810930, 0.262365, 0.336472])
            self.assertTrue(tk.allclose(losses_none, expected_none))

            # Test with reduction 'mean'
            losses_mean = nn.losses.binary_cross_entropy(
                logits, targets, reduction="mean"
            )
            expected_mean = tk.mean(expected_none)
            self.assertTrue(tk.allclose(losses_mean, expected_mean))

            # Test with reduction 'sum'
            losses_sum = nn.losses.binary_cross_entropy(
                logits, targets, reduction="sum"
            )
            expected_sum = tk.sum(expected_none)
            self.assertTrue(tk.allclose(losses_sum, expected_sum))

            # With weights, no label smoothing
            weights = tk.array([1.0, 2.0, 1.0, 2.0])
            expected = tk.array([0.747215, 1.62186, 0.262365, 0.672944])
            loss = nn.losses.binary_cross_entropy(
                logits, targets, weights=weights, reduction="none"
            )
            self.assertTrue(tk.allclose(loss, expected))

        def _test_probs_as_inputs():
            probs = tk.array([0.5, 0.6, 0.7, 0.8])
            targets = tk.array([0, 0, 1, 1])

            # Test with reduction 'none'
            losses_none = nn.losses.binary_cross_entropy(
                probs, targets, with_logits=False, reduction="none"
            )
            expected_none = tk.array([0.693147, 0.916291, 0.356675, 0.223144])
            self.assertTrue(tk.allclose(losses_none, expected_none))

            # Test with reduction 'mean'
            losses_mean = nn.losses.binary_cross_entropy(
                probs, targets, with_logits=False, reduction="mean"
            )
            expected_mean = tk.mean(expected_none)
            self.assertTrue(tk.allclose(losses_mean, expected_mean))

            # Test with reduction 'sum'
            losses_sum = nn.losses.binary_cross_entropy(
                probs, targets, with_logits=False, reduction="sum"
            )
            expected_sum = tk.sum(expected_none)
            self.assertTrue(tk.allclose(losses_sum, expected_sum))

        def _test_tiny_probs_as_inputs():
            TINY_PROB = 1e-59
            probs = tk.array([0, TINY_PROB, 1 - TINY_PROB, 1])
            targets = tk.array([0, 0, 1, 1])

            losses_none = nn.losses.binary_cross_entropy(
                probs, targets, with_logits=False, reduction="none"
            )
            expected_none = tk.array([0.0, TINY_PROB, TINY_PROB, 0.0])
            self.assertTrue(tk.allclose(losses_none, expected_none))

            # Test with reduction 'mean'
            losses_mean = nn.losses.binary_cross_entropy(
                probs, targets, with_logits=False, reduction="mean"
            )
            expected_mean = tk.mean(expected_none)
            self.assertTrue(tk.allclose(losses_mean, expected_mean))

            # Test with reduction 'sum'
            losses_sum = nn.losses.binary_cross_entropy(
                probs, targets, with_logits=False, reduction="sum"
            )
            expected_sum = tk.sum(expected_none)
            self.assertTrue(tk.allclose(losses_sum, expected_sum))

        _test_logits_as_inputs()
        _test_probs_as_inputs()
        _test_tiny_probs_as_inputs()

    def test_l1_loss(self):
        predictions = tk.array([0.5, 0.2, 0.9, 0.0])
        targets = tk.array([0.5, 0.2, 0.9, 0.0])

        # Expected result
        expected_none = tk.array([0, 0, 0, 0]).astype(tk.float32)
        expected_sum = tk.sum(expected_none)
        expected_mean = tk.mean(expected_none)

        losses = nn.losses.l1_loss(predictions, targets, reduction="none")
        self.assertTrue(
            tk.array_equal(losses, expected_none),
            "Test failed for l1_loss --reduction='none'",
        )

        losses = nn.losses.l1_loss(predictions, targets, reduction="sum")
        self.assertTrue(tk.array_equal(losses, expected_sum))

        losses = nn.losses.l1_loss(predictions, targets, reduction="mean")
        self.assertTrue(tk.array_equal(losses, expected_mean))

    def test_mse_loss(self):
        predictions = tk.array([0.5, 0.2, 0.9, 0.0])
        targets = tk.array([0.7, 0.1, 0.8, 0.2])

        expected_none = tk.array([0.04, 0.01, 0.01, 0.04])
        expected_mean = tk.mean(expected_none)
        expected_sum = tk.sum(expected_none)

        # Test with reduction 'none'
        losses_none = nn.losses.mse_loss(predictions, targets, reduction="none")
        self.assertTrue(
            np.allclose(losses_none, expected_none, 1e-5),
            "Test case failed for mse_loss --reduction='none'",
        )

        # Test with reduction 'mean'
        losses_mean = nn.losses.mse_loss(predictions, targets, reduction="mean")
        self.assertEqual(
            losses_mean,
            expected_mean,
            "Test case failed for mse_loss --reduction='mean'",
        )

        # Test with reduction 'sum'
        losses_sum = nn.losses.mse_loss(predictions, targets, reduction="sum")
        self.assertEqual(
            losses_sum, expected_sum, "Test case failed for mse_loss --reduction='sum'"
        )

    def test_smooth_l1_loss(self):
        predictions = tk.array([1.5, 2.5, 0.5, 3.5])
        targets = tk.array([1.0, 2.0, 0.5, 2.5])
        beta = 1.0

        # Expected results
        expected_none = tk.array([0.125, 0.125, 0.0, 0.5])
        expected_sum = tk.sum(expected_none)
        expected_mean = tk.mean(expected_none)

        # Test with reduction 'none'
        loss_none = nn.losses.smooth_l1_loss(
            predictions, targets, beta, reduction="none"
        )
        self.assertTrue(
            tk.array_equal(loss_none, expected_none),
            "Test case failed for smooth_l1_loss --reduction='none'",
        )

        # Test with reduction 'sum'
        loss_sum = nn.losses.smooth_l1_loss(predictions, targets, beta, reduction="sum")
        self.assertEqual(
            loss_sum,
            expected_sum,
            "Test case failed for smooth_l1_loss --reduction='sum'",
        )

        # Test with reduction 'mean'
        loss_mean = nn.losses.smooth_l1_loss(
            predictions, targets, beta, reduction="mean"
        )
        self.assertEqual(
            loss_mean,
            expected_mean,
            "Test case failed for smooth_l1_loss --reduction='mean'",
        )

    def test_nll_loss(self):
        logits = tk.array([[0.0, -float("inf")], [-float("inf"), 0.0]])
        targets = tk.array([0, 1])

        # Test with reduction 'none'
        losses_none = nn.losses.nll_loss(logits, targets, reduction="none")
        expected_none = tk.array([0.0, 0.0])
        self.assertTrue(tk.array_equal(losses_none, expected_none))

        # Test with reduction 'mean'
        losses_mean = nn.losses.nll_loss(logits, targets, reduction="mean")
        expected_mean = tk.mean(expected_none)
        self.assertEqual(losses_mean, expected_mean)

        # Test with reduction 'sum'
        losses_sum = nn.losses.nll_loss(logits, targets, reduction="sum")
        expected_sum = tk.sum(expected_none)
        self.assertEqual(losses_sum, expected_sum)

        # Test a different axis
        logits = tk.log(tk.softmax(tk.random.normal((4, 8)), axis=-1))
        targets = tk.array([1, 2, 3, 0])
        loss = nn.losses.nll_loss(logits.T, targets, axis=0)
        expected = nn.losses.nll_loss(logits, targets, axis=-1)
        self.assertTrue(tk.allclose(loss, expected))

        # Test a middle axis on a higher rank input
        logits = tk.log(tk.softmax(tk.random.normal((2, 5, 3)), axis=1))
        targets = tk.array([[1, 4, 0], [2, 3, 1]])
        loss = nn.losses.nll_loss(logits, targets, axis=1)
        logits_np = np.array(logits)
        targets_np = np.array(targets)
        expected = np.zeros((2, 3))
        for b in range(2):
            for k in range(3):
                expected[b, k] = -logits_np[b, targets_np[b, k], k]
        self.assertEqual(loss.shape, (2, 3))
        self.assertTrue(np.allclose(np.array(loss), expected))

    def test_gaussian_nll_loss(self):
        inputs = tk.array([[0.1, 0.2], [0.3, 0.4]])
        targets = tk.array([[0.2, 0.1], [0.1, 0.2]])
        vars = tk.array([[0.1, 0.2], [0.3, 0.4]])

        # Test with reduction 'none', full=False
        losses_none = nn.losses.gaussian_nll_loss(
            inputs, targets, vars, reduction="none"
        )
        expected_none = tk.array([[-1.101293, -0.779719], [-0.535320, -0.408145]])
        self.assertTrue(tk.allclose(losses_none, expected_none))

        # Test with reduction 'mean', full=False
        losses_mean = nn.losses.gaussian_nll_loss(
            inputs, targets, vars, reduction="mean"
        )
        expected_mean = tk.mean(expected_none)
        self.assertTrue(tk.allclose(losses_mean, expected_mean))

        # Test with reduction 'sum', full=False
        losses_sum = nn.losses.gaussian_nll_loss(inputs, targets, vars, reduction="sum")
        expected_sum = tk.sum(expected_none)
        self.assertTrue(tk.allclose(losses_sum, expected_sum))

        # Test with reduction='none', full=True
        losses_none_full = nn.losses.gaussian_nll_loss(
            inputs, targets, vars, full=True, reduction="none"
        )
        expected_none_full = tk.array([[-0.182354, 0.139220], [0.383619, 0.510793]])
        self.assertTrue(tk.allclose(losses_none_full, expected_none_full))

        # Test with reduction='mean', full=True
        losses_mean_full = nn.losses.gaussian_nll_loss(
            inputs, targets, vars, full=True, reduction="mean"
        )
        expected_mean_full = tk.mean(expected_none_full)
        self.assertTrue(tk.allclose(losses_mean_full, expected_mean_full))

        # Test with reduction='sum', full=True
        losses_sum_full = nn.losses.gaussian_nll_loss(
            inputs, targets, vars, full=True, reduction="sum"
        )
        expected_sum_full = tk.sum(expected_none_full)
        self.assertTrue(tk.allclose(losses_sum_full, expected_sum_full))

        # The default reduction is "mean" (matches the documented default): a
        # scalar equal to the explicit mean reduction, not the per-element array.
        losses_default = nn.losses.gaussian_nll_loss(inputs, targets, vars)
        self.assertEqual(losses_default.ndim, 0)
        self.assertTrue(tk.allclose(losses_default, expected_mean))

    def test_kl_div_loss(self):
        p_logits = tk.log(tk.array([[0.5, 0.5], [0.8, 0.2]]))
        q_logits = tk.log(tk.array([[0.5, 0.5], [0.2, 0.8]]))

        # Test with reduction 'none'
        losses_none = nn.losses.kl_div_loss(p_logits, q_logits, reduction="none")
        expected_none = tk.array([0.0, 0.831777])
        self.assertTrue(tk.allclose(losses_none, expected_none))

        # The documented formula must match the implementation: the sum reduces
        # the elementwise product over `axis`, so the multiply has to be inside
        # the sum (parenthesization matters).
        documented = (tk.exp(q_logits) * (q_logits - p_logits)).sum(axis=-1)
        self.assertTrue(tk.allclose(losses_none, documented))

        # Test with reduction 'mean'
        losses_mean = nn.losses.kl_div_loss(p_logits, q_logits, reduction="mean")
        expected_mean = tk.mean(expected_none)
        self.assertTrue(tk.allclose(losses_mean, expected_mean))

        # Test with reduction 'sum'
        losses_sum = nn.losses.kl_div_loss(p_logits, q_logits, reduction="sum")
        expected_sum = tk.sum(expected_none)
        self.assertTrue(tk.allclose(losses_sum, expected_sum))

    def test_triplet_loss(self):
        anchors = tk.array([[1, 2, 3], [1, 2, 3]])
        positives = tk.array([[4, 5, 6], [0, -1, 2]])
        negatives = tk.array([[7, 8, 9], [3, 2, 3]])

        # Test with reduction 'none'
        losses_none = nn.losses.triplet_loss(
            anchors, positives, negatives, reduction="none"
        )
        expected_none = tk.array([0, 2.31662])
        self.assertTrue(tk.allclose(losses_none, expected_none))

        # Test with reduction 'mean'
        losses_mean = nn.losses.triplet_loss(
            anchors, positives, negatives, reduction="mean"
        )
        expected_mean = tk.mean(expected_none)
        self.assertTrue(tk.allclose(losses_mean, expected_mean))

        # Test with reduction 'sum'
        losses_sum = nn.losses.triplet_loss(
            anchors, positives, negatives, reduction="sum"
        )
        expected_sum = tk.sum(expected_none)
        self.assertTrue(tk.allclose(losses_sum, expected_sum))

        # Test with non-default 'p' norm degrees
        anchors = tk.array([[0.0, 0.0]])
        positives = tk.array([[2.0, 0.0]])
        negatives = tk.array([[0.0, 3.0]])

        losses_p1 = nn.losses.triplet_loss(
            anchors, positives, negatives, p=1, reduction="none"
        )
        expected_p1 = tk.array([0.0])
        self.assertTrue(tk.allclose(losses_p1, expected_p1))

        losses_p3 = nn.losses.triplet_loss(
            anchors, positives, negatives, p=3, reduction="none"
        )
        expected_p3 = tk.array([0.0])
        self.assertTrue(tk.allclose(losses_p3, expected_p3))

    def test_hinge_loss(self):
        inputs = tk.ones((2, 4))
        targets = tk.zeros((2, 4))
        loss = nn.losses.hinge_loss(inputs, targets, reduction="mean")
        self.assertEqual(loss, 1.0)

    def test_huber_loss(self):
        inputs = tk.ones((2, 4))
        targets = tk.zeros((2, 4))
        loss = nn.losses.huber_loss(inputs, targets, reduction="mean")
        self.assertEqual(loss, 0.5)

    def test_log_cosh_loss(self):
        inputs = tk.ones((2, 4))
        targets = tk.zeros((2, 4))
        loss = nn.losses.log_cosh_loss(inputs, targets, reduction="mean")
        self.assertAlmostEqual(loss.item(), 0.433781, places=6)

    def test_cosine_similarity_loss(self):
        embeddings1 = tk.array([[0.5, 0.5, 0.2, 0.9], [0.1, 0.3, 0.5, 0.5]])
        embeddings2 = tk.array([[0.6, 0.4, 0.3, 0.8], [0.2, 0.5, 0.6, 0.4]])

        # Test with reduction 'none'
        losses_none = nn.losses.cosine_similarity_loss(
            embeddings1, embeddings2, reduction="none"
        )
        expected_none = tk.array([0.985344, 0.961074])
        self.assertTrue(tk.allclose(losses_none, expected_none))

        # Test with reduction 'mean'
        losses_mean = nn.losses.cosine_similarity_loss(
            embeddings1, embeddings2, reduction="mean"
        )
        expected_mean = tk.mean(expected_none)
        self.assertTrue(tk.allclose(losses_mean, expected_mean))

        # Test with reduction 'sum'
        losses_sum = nn.losses.cosine_similarity_loss(
            embeddings1, embeddings2, reduction="sum"
        )
        expected_sum = tk.sum(expected_none)
        self.assertTrue(tk.allclose(losses_sum, expected_sum))

    def test_margin_ranking_loss(self):
        inputs1 = tk.array([-0.573409, -0.765166, -0.0638])
        inputs2 = tk.array([0.75596, 0.225763, 0.256995])
        targets = tk.array([1, 1, -1])

        # Test with no margin
        losses = nn.losses.margin_ranking_loss(
            inputs1, inputs2, targets, reduction="none"
        )
        expected = tk.array([1.329369, 0.990929, 0.0])
        self.assertTrue(tk.allclose(losses, expected))

        # Test with margin
        losses = nn.losses.margin_ranking_loss(
            inputs1, inputs2, targets, margin=0.5, reduction="none"
        )
        expected = tk.array([1.829369, 1.490929, 0.179205])
        self.assertTrue(tk.allclose(losses, expected))


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
