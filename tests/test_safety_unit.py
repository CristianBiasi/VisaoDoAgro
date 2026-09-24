"""Limites e memória temporal das regras de segurança, sem câmera ou IA."""
import unittest

from proximity import ProximityEstimator
from risk import CollisionRiskEngine, Risk


class RiskBoundaryTests(unittest.TestCase):
    def candidate(self, proximity, approach=0, overlap=0, confidence=.5):
        return CollisionRiskEngine().candidate(
            proximity, approach, overlap, confidence, 32, 58, 78, 8,
        )

    def test_risk_threshold_boundaries(self):
        for score, expected in (
            (0, Risk.SAFE), (11.9, Risk.SAFE), (12, Risk.LOW),
            (31.9, Risk.LOW), (32, Risk.MEDIUM), (57.9, Risk.MEDIUM),
            (58, Risk.HIGH), (100, Risk.HIGH),
        ):
            with self.subTest(proximity=score):
                self.assertEqual(self.candidate(score), expected)

    def test_critical_requires_proximity_corridor_and_approach(self):
        # Perigo combinado = 97,6: abaixo da outra condição crítica (102).
        self.assertEqual(self.candidate(78, 8, .35), Risk.CRITICAL)
        for score, approach, overlap in ((77.9, 8, .35), (78, 7.9, .35), (78, 8, .349)):
            with self.subTest(proximity=score, approach=approach, overlap=overlap):
                self.assertEqual(self.candidate(score, approach, overlap), Risk.HIGH)

    def test_combined_danger_can_be_critical_without_approach(self):
        self.assertEqual(self.candidate(78, overlap=1), Risk.CRITICAL)
        self.assertEqual(self.candidate(77.9, overlap=1), Risk.HIGH)

    def test_receding_object_and_low_confidence_do_not_subtract_danger(self):
        self.assertEqual(self.candidate(58, approach=-20, confidence=.1), Risk.HIGH)
        self.assertEqual(self.candidate(30, confidence=1), Risk.MEDIUM)

    def test_confirmation_is_independent_for_each_object(self):
        engine = CollisionRiskEngine()
        self.assertEqual(engine.update(1, Risk.HIGH, 2), Risk.SAFE)
        self.assertEqual(engine.update(2, Risk.HIGH, 2), Risk.SAFE)
        self.assertEqual(engine.update(1, Risk.HIGH, 2), Risk.HIGH)
        self.assertEqual(engine.update(2, Risk.SAFE, 2), Risk.SAFE)
        self.assertEqual(engine.update(2, Risk.HIGH, 2), Risk.SAFE)

    def test_deescalation_needs_one_extra_consecutive_frame(self):
        engine = CollisionRiskEngine()
        self.assertEqual([engine.update(1, Risk.CRITICAL, 2) for _ in range(2)],
                         [Risk.SAFE, Risk.CRITICAL])
        self.assertEqual([engine.update(1, Risk.LOW, 2) for _ in range(3)],
                         [Risk.CRITICAL, Risk.CRITICAL, Risk.LOW])

    def test_corridor_overlap_uses_object_area(self):
        engine = CollisionRiskEngine()
        for bbox, expected in (([20, 40, 40, 60], .75), ([0, 0, 100, 100], .25),
                               ([0, 0, 25, 25], 0), ([50, 50, 50, 50], 0)):
            with self.subTest(bbox=bbox):
                self.assertAlmostEqual(engine.corridor_overlap(bbox, 100, 100, .5, .5), expected)


class ProximityUnitTests(unittest.TestCase):
    def test_smoothing_and_approach_use_elapsed_time(self):
        estimator = ProximityEstimator(smoothing=.25)
        first = estimator.update(7, [0, 0, 10, 10], 100, 100, 10)
        second = estimator.update(7, [0, 0, 30, 30], 100, 100, 12)
        self.assertEqual((first.score, first.approach_rate), (15, 0))
        # Score bruto 45; suavizado 15 + .25 * (45 - 15) = 22,5.
        self.assertEqual((second.score, second.approach_rate), (22.5, 3.8))

    def test_same_relative_box_has_same_score_at_different_resolutions(self):
        estimator = ProximityEstimator()
        small = estimator.update(1, [10, 10, 30, 30], 100, 100, 1)
        large = estimator.update(2, [20, 20, 60, 60], 200, 200, 1)
        self.assertEqual(small.score, 30)
        self.assertEqual(small, large)

    def test_history_window_excludes_old_samples(self):
        estimator = ProximityEstimator(history_size=2, smoothing=1)
        estimator.update(1, [0, 0, 10, 10], 100, 100, 1)
        estimator.update(1, [0, 0, 20, 20], 100, 100, 2)
        result = estimator.update(1, [0, 0, 40, 40], 100, 100, 3)
        self.assertEqual(result.approach_rate, 30)

    def test_reappearing_id_starts_without_previous_approach(self):
        estimator = ProximityEstimator(smoothing=1)
        for track_id in (1, 2):
            estimator.update(track_id, [0, 0, 10, 10], 100, 100, 1)
        estimator.forget_missing({2})
        reappeared = estimator.update(1, [0, 0, 40, 40], 100, 100, 2)
        retained = estimator.update(2, [0, 0, 20, 20], 100, 100, 2)
        self.assertEqual((reappeared.score, reappeared.approach_rate), (60, 0))
        self.assertEqual(retained.approach_rate, 15)

    def test_equal_timestamps_produce_finite_rate(self):
        estimator = ProximityEstimator(smoothing=1)
        estimator.update(1, [0, 0, 10, 10], 100, 100, 1)
        result = estimator.update(1, [0, 0, 20, 20], 100, 100, 1)
        self.assertEqual(result.approach_rate, 15000)
