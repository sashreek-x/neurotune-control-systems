import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import app

class NeuroTuneTests(unittest.TestCase):
    def test_parse_coefficients(self):
        self.assertEqual(app.parse_coeffs("1, 2, 3", "test").tolist(), [1.0,2.0,3.0])

    def test_closed_loop_second_order(self):
        num, den = app.get_plant({"num":"1","den":"1,3,2"})
        cn, cd = app.closed_loop(num, den, 3, 1, .2)
        self.assertTrue(len(cd) >= 3)
        self.assertTrue(all(abs(x) < 1e9 for x in cd))

    def test_simulation_metrics(self):
        result = app.simulate({"num":"1","den":"1,3,2","kp":3,"ki":1,"kd":.2,"duration":5})
        self.assertIn("metrics", result)
        self.assertEqual(len(result["time"]), len(result["closed_loop"]))
        self.assertEqual(len(result["poles"]), len(result["closed_loop_tf"]["den"])-1)

    def test_feedback_reduction(self):
        num=app.parse_coeffs("1","num"); den=app.parse_coeffs("1,3,2","den")
        reduced=app.poly_add(den,num)
        self.assertEqual(reduced.tolist(), [1.0,3.0,3.0])

    def test_flask_health(self):
        client=app.app.test_client()
        response=client.get("/health")
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json["status"],"ok")

    def test_simulation_endpoint(self):
        client=app.app.test_client()
        response=client.post("/api/simulate",json={"num":"1","den":"1,3,2","kp":3,"ki":1,"kd":.2,"duration":5})
        self.assertEqual(response.status_code,200)
        self.assertIn("closed_loop",response.json)

if __name__ == "__main__":
    unittest.main()
