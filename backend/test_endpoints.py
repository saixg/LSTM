import unittest
from backend.app.routers.explainer import trace_explainer, ExplainerTraceRequest
from backend.app.routers.nextword import predict_nextword, NextWordRequest
from backend.app.routers.anomaly import get_anomaly_samples, detect_anomaly, AnomalyDetectRequest
from backend.app.routers.sentiment import classify_sentiment, SentimentRequest

class TestNeuroSeqDirect(unittest.TestCase):
    def test_explainer_trace(self):
        req = ExplainerTraceRequest(text="the neural network learns patterns")
        res = trace_explainer(req)
        self.assertGreater(len(res.steps), 0)
        for step in res.steps:
            self.assertTrue(0.0 <= step.forget <= 1.0)
            self.assertTrue(0.0 <= step.input <= 1.0)
            self.assertTrue(0.0 <= step.output <= 1.0)
            self.assertGreaterEqual(step.cell_state_norm, 0.0)

    def test_nextword_predict(self):
        req = NextWordRequest(text="artificial intelligence will", level="word")
        res = predict_nextword(req)
        self.assertEqual(len(res.top5), 5)
        self.assertGreater(len(res.steps), 0)
        for step in res.steps:
            self.assertTrue(0.0 <= step.forget <= 1.0)
            self.assertTrue(0.0 <= step.input <= 1.0)
            self.assertTrue(0.0 <= step.output <= 1.0)

    def test_anomaly_samples_and_detect(self):
        samples = get_anomaly_samples()
        self.assertGreater(len(samples), 0)
        sample_id = samples[0]["id"]

        req = AnomalyDetectRequest(sequence_id=sample_id, inject_anomaly=True)
        res = detect_anomaly(req)
        self.assertEqual(len(res.input), len(res.reconstruction))
        self.assertEqual(len(res.error), len(res.reconstruction))
        self.assertGreater(len(res.steps), 0)

    def test_sentiment_classify(self):
        req = SentimentRequest(text="the movie was amazing and fantastic")
        res = classify_sentiment(req)
        self.assertEqual(res.final_label, "Positive")
        self.assertGreater(len(res.steps), 0)

if __name__ == "__main__":
    unittest.main()
