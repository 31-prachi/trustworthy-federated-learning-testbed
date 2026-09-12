import unittest
import numpy as np
from fl_testbed.core import make_clients, fedavg, metrics, membership_inference_auc


class TestCore(unittest.TestCase):
    def test_metrics_perfect(self):
        result = metrics(np.array([0, 1, 1]), np.array([0, 1, 1]))
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["f1"], 1.0)

    def test_fedavg_reproducible(self):
        clients = make_clients(n_clients=4, samples_per_client=60, features=5, seed=7)
        first, _ = fedavg(clients, features=5, rounds=5, seed=7)
        second, _ = fedavg(clients, features=5, rounds=5, seed=7)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first["accuracy"], 0.5)

    def test_client_data_shapes(self):
        clients = make_clients(n_clients=3, samples_per_client=40, features=4, seed=3)
        self.assertEqual(len(clients), 3)
        self.assertTrue(all(client.x_train.shape == (30, 4) for client in clients))
        self.assertTrue(all(client.x_test.shape == (10, 4) for client in clients))

    def test_privacy_metadata_and_attack_range(self):
        clients = make_clients(n_clients=4, samples_per_client=80, features=5, seed=9)
        result, _ = fedavg(
            clients, 5, rounds=4, seed=9, noise_multiplier=0.1,
            secure_aggregation=True,
        )
        self.assertTrue(result["privacy_mechanism"]["secure_aggregation_simulated"])
        self.assertFalse(result["privacy_mechanism"]["formal_privacy_accounting"])
        self.assertTrue(0.0 <= result["membership_attack_auc"] <= 1.0)

    def test_membership_attack_auc_is_valid(self):
        clients = make_clients(n_clients=3, samples_per_client=60, features=4, seed=3)
        auc = membership_inference_auc(clients, np.zeros(4), 0.0)
        self.assertTrue(0.0 <= auc <= 1.0)


if __name__ == "__main__":
    unittest.main()
