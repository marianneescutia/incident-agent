import unittest

from finetuning.reward_functions import concise_json_reward


class RewardTests(unittest.TestCase):
    def test_rewards_concise_exact_json(self):
        output = (
            '{"immediate_action":"Isolate the affected service now.",'
            '"short_term_mitigation":"Route traffic to a healthy replica.",'
            '"long_term_prevention":"Add automated failover tests and alerts."}'
        )
        self.assertEqual(concise_json_reward([output]), [1.5])

    def test_rejects_explanation_outside_json_bonus(self):
        output = (
            'Here is the answer: '
            '{"immediate_action":"Isolate service.",'
            '"short_term_mitigation":"Route traffic.",'
            '"long_term_prevention":"Add failover tests."}'
        )
        self.assertEqual(concise_json_reward([output]), [1.0])


if __name__ == "__main__":
    unittest.main()
