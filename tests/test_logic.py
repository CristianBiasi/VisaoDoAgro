import unittest
from proximity import ProximityEstimator
from risk import CollisionRiskEngine, Risk

class LogicTests(unittest.TestCase):
    def test_proximity_increases_with_bbox(self):
        estimator = ProximityEstimator(smoothing=1)
        small = estimator.update(1, [0, 0, 100, 100], 1000, 1000, 1).score
        large = estimator.update(1, [0, 0, 500, 500], 1000, 1000, 2).score
        self.assertGreater(large, small)
        self.assertGreater(estimator.update(1, [0, 0, 500, 500], 1000, 1000, 3).approach_rate, 0)

    def test_corridor_intersection(self):
        engine = CollisionRiskEngine()
        self.assertGreater(engine.corridor_overlap([400, 400, 600, 600], 1000, 1000, .5, .6), .9)
        self.assertEqual(engine.corridor_overlap([0, 0, 100, 100], 1000, 1000, .5, .6), 0)

    def test_hysteresis_requires_confirmation(self):
        engine = CollisionRiskEngine()
        self.assertEqual(engine.update(1, Risk.HIGH, 3), Risk.SAFE)
        self.assertEqual(engine.update(1, Risk.HIGH, 3), Risk.SAFE)
        self.assertEqual(engine.update(1, Risk.HIGH, 3), Risk.HIGH)

if __name__ == '__main__': unittest.main()
