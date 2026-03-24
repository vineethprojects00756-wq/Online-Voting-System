import importlib
import sys
import types
import unittest


def load_main_module():
    sys.modules.pop("main", None)

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.array = lambda value: value
    sys.modules["numpy"] = numpy_stub

    cv2_stub = types.ModuleType("cv2")
    cv2_stub.COLOR_BGR2GRAY = 0
    cv2_stub.FONT_HERSHEY_SIMPLEX = 0
    cv2_stub.error = Exception
    cv2_stub.data = types.SimpleNamespace(haarcascades="")

    class DummyCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return []

    class DummyRecognizer:
        def train(self, *_args, **_kwargs):
            return None

        def predict(self, *_args, **_kwargs):
            return 0, 100

    class DummyCamera:
        def isOpened(self):
            return False

        def read(self):
            return False, None

        def release(self):
            return None

    cv2_stub.CascadeClassifier = lambda *_args, **_kwargs: DummyCascade()
    cv2_stub.cvtColor = lambda image, *_args, **_kwargs: image
    cv2_stub.resize = lambda image, *_args, **_kwargs: image
    cv2_stub.imread = lambda *_args, **_kwargs: object()
    cv2_stub.imwrite = lambda *_args, **_kwargs: True
    cv2_stub.flip = lambda image, *_args, **_kwargs: image
    cv2_stub.getRotationMatrix2D = lambda *_args, **_kwargs: object()
    cv2_stub.warpAffine = lambda image, *_args, **_kwargs: image
    cv2_stub.convertScaleAbs = lambda image, *_args, **_kwargs: image
    cv2_stub.VideoCapture = lambda *_args, **_kwargs: DummyCamera()
    cv2_stub.imencode = lambda *_args, **_kwargs: (True, b"")
    cv2_stub.rectangle = lambda *_args, **_kwargs: None
    cv2_stub.putText = lambda *_args, **_kwargs: None
    cv2_stub.face = types.SimpleNamespace(LBPHFaceRecognizer_create=lambda: DummyRecognizer())
    sys.modules["cv2"] = cv2_stub

    mongo_stub = types.ModuleType("mongo_db")
    mongo_stub.create_user = lambda *_args, **_kwargs: True
    mongo_stub.find_user_by_credentials = (
        lambda aadhar, password: {"aadhar": aadhar}
        if aadhar == "123456789012" and password == "secret"
        else None
    )
    mongo_stub.get_face_records = lambda: []
    mongo_stub.get_system_metrics = lambda: {
        "registered_citizens": 0,
        "face_datasets": 0,
        "trained_models": 0,
        "votes_stored": 0,
        "audit_events": 0,
        "active_model": None,
        "latest_audit": None,
        "device_logs": 0,
        "votes_cast": 0,
        "remaining_voters": 0,
    }
    mongo_stub.initialize_database = lambda: None
    mongo_stub.get_registration_overview = lambda limit=5: []
    mongo_stub.get_available_constituencies = lambda: ["Vizag"]
    mongo_stub.get_all_voters = lambda: [
        {
            "voter_id": "V12345",
            "name": "Test Voter",
            "email": "test@gmail.com",
            "phone": "9999999999",
            "constituency": "Vizag",
            "registered": True,
            "has_voted": False,
            "eligibility_status": "pending_verification",
        }
    ]
    mongo_stub.get_candidate_list = lambda: [
        {"candidate_id": "CAND-A", "name": "Candidate A", "party": "Reform Alliance", "constituency": "Vizag"},
        {"candidate_id": "CAND-B", "name": "Candidate B", "party": "People's Front", "constituency": "Vizag"},
        {"candidate_id": "CAND-C", "name": "Candidate C", "party": "Unity Movement", "constituency": "Vizag"},
    ]
    mongo_stub.get_recent_device_logs = lambda limit=10: []
    mongo_stub.get_recent_votes = lambda limit=10: []
    mongo_stub.get_vote_results = lambda: [
        {"candidate_id": "CAND-A", "name": "Candidate A", "party": "Reform Alliance", "constituency": "Vizag", "votes": 1520},
        {"candidate_id": "CAND-B", "name": "Candidate B", "party": "People's Front", "constituency": "Vizag", "votes": 1470},
    ]
    mongo_stub.get_voter_by_aadhar = lambda aadhar: {
        "aadhar": aadhar,
        "voter_id": "V12345",
        "name": "Test Voter",
        "constituency": "Vizag",
        "registered": True,
        "has_voted": False,
    }
    mongo_stub.get_voter_by_voter_id = lambda voter_id: {
        "aadhar": "123456789012",
        "voter_id": voter_id,
        "name": "Test Voter",
        "constituency": "Vizag",
        "registered": True,
        "has_voted": False,
    }
    mongo_stub.check_voting_status = lambda voter_id: {
        "allowed": True,
        "message": "Vote allowed - voter is eligible and has not voted yet.",
        "voter": mongo_stub.get_voter_by_voter_id(voter_id),
    }
    mongo_stub.store_vote = lambda voter_id, vote_id, constituency, encrypted_candidate_id: {
        "stored": True,
        "message": "Vote stored securely and voter status updated.",
        "vote": {
            "vote_id": vote_id,
            "constituency": constituency,
            "candidate_id": encrypted_candidate_id,
        },
    }
    mongo_stub.add_candidate = lambda *_args, **_kwargs: None
    mongo_stub.log_audit_event = lambda *_args, **_kwargs: None
    mongo_stub.log_device_activity = lambda *_args, **_kwargs: None
    mongo_stub.remove_candidate = lambda *_args, **_kwargs: True
    mongo_stub.reset_election = lambda: {"reset_voters": 1, "deleted_votes": 0}
    mongo_stub.update_candidate = lambda *_args, **_kwargs: True
    mongo_stub.update_model_status = lambda *_args, **_kwargs: None
    mongo_stub.update_voter_record = lambda *_args, **_kwargs: True
    mongo_stub.delete_voter_record = lambda *_args, **_kwargs: True
    sys.modules["mongo_db"] = mongo_stub

    return importlib.import_module("main")


class AppFlowTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main_module()
        cls.main.app.config["TESTING"] = True

    def setUp(self):
        self.client = self.main.app.test_client()

    def test_home_page_renders_primary_navigation(self):
        response = self.client.get("/")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Register as Voter", body)
        self.assertIn("Voter Login", body)
        self.assertIn("Voting journey", body)
        self.assertIn("Admin Login", body)
        self.assertIn("Recognition Model Options", body)

    def test_health_route_reports_runtime_status(self):
        response = self.client.get("/health")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertIn("database_available", payload)
        self.assertIn("server_camera_enabled", payload)

    def test_voter_dashboard_requires_login(self):
        response = self.client.get("/voter/dashboard")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/login"))

    def test_admin_dashboard_requires_login(self):
        response = self.client.get("/admin/dashboard")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/admin/login"))

    def test_otp_page_requires_login_session(self):
        response = self.client.get("/otp_verification")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/login"))

    def test_face_verification_requires_verified_session(self):
        response = self.client.get("/face_verification")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/login"))

    def test_video_feed_requires_verified_session(self):
        response = self.client.get("/video_feed")
        self.assertEqual(response.status_code, 403)

    def test_status_route_returns_authentication_payload(self):
        response = self.client.get("/status")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("status", payload)
        self.assertIn("decision", payload)
        self.assertIn("face_match", payload)

    def test_admin_dashboard_data_returns_live_payload(self):
        response = self.client.get("/admin/dashboard/data")
        payload = response.get_json()

        self.assertEqual(response.status_code, 403)
        self.assertEqual(payload["error"], "Admin authentication required.")

    def test_admin_login_success_opens_dashboard(self):
        response = self.client.post(
            "/admin/login",
            data={"username": "admin", "password": "admin123"},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/admin/dashboard"))

        with self.client.session_transaction() as session_data:
            self.assertTrue(session_data["admin_authenticated"])
            self.assertEqual(session_data["admin_username"], "admin")

    def test_admin_dashboard_renders_architecture_monitoring_for_authenticated_admin(self):
        with self.client.session_transaction() as session_data:
            session_data["admin_authenticated"] = True
            session_data["admin_username"] = "admin"

        response = self.client.get("/admin/dashboard")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Monitor election operations.", body)
        self.assertIn("Manage Candidates", body)
        self.assertIn("Election Control", body)
        self.assertIn("Live Result Snapshot", body)

    def test_admin_dashboard_data_returns_live_payload_for_authenticated_admin(self):
        with self.client.session_transaction() as session_data:
            session_data["admin_authenticated"] = True
            session_data["admin_username"] = "admin"

        response = self.client.get("/admin/dashboard/data")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("metrics", payload)
        self.assertIn("recent_voters", payload)
        self.assertIn("recent_votes", payload)
        self.assertIn("candidates", payload)

    def test_vote_route_requires_verified_vote_access(self):
        response = self.client.get("/vote")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/face", response.headers["Location"])

    def test_login_success_starts_otp_flow(self):
        response = self.client.post(
            "/login",
            data={"aadhar": "123456789012", "password": "secret"},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/otp_verification"))

        with self.client.session_transaction() as session_data:
            self.assertEqual(session_data["aadhar"], "123456789012")
            self.assertFalse(session_data["otp_verified"])
            self.assertEqual(len(session_data["otp"]), 6)

    def test_otp_success_advances_to_face_verification(self):
        with self.client.session_transaction() as session_data:
            session_data["aadhar"] = "123456789012"
            session_data["otp"] = "654321"
            session_data["otp_verified"] = False

        response = self.client.post(
            "/otp_verification",
            data={"otp": "654321"},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/voter/dashboard"))

        with self.client.session_transaction() as session_data:
            self.assertTrue(session_data["otp_verified"])
            self.assertNotIn("otp", session_data)

    def test_face_verification_renders_for_verified_session(self):
        with self.client.session_transaction() as session_data:
            session_data["aadhar"] = "123456789012"
            session_data["otp_verified"] = True

        response = self.client.get("/face_verification")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Face Verification", response.get_data(as_text=True))
        self.assertIn("Capture & Verify", response.get_data(as_text=True))
        self.assertIn("Verification Status", response.get_data(as_text=True))

    def test_vote_page_renders_candidates_when_access_is_allowed(self):
        with self.client.session_transaction() as session_data:
            session_data["otp_verified"] = True

        self.main.current_authentication = {
            "status": "Vote allowed - voter is eligible and has not voted yet.",
            "decision": "allow",
            "face_match": True,
            "voter_id": "V12345",
            "name": "Test Voter",
            "constituency": "Vizag",
            "has_voted": False,
        }

        response = self.client.get("/vote")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Cast Your Vote", body)
        self.assertIn("Candidate A", body)
        self.assertIn("Candidate B", body)
        self.assertIn("Candidate C", body)

    def test_verify_face_returns_vote_redirect_for_eligible_voter(self):
        with self.client.session_transaction() as session_data:
            session_data["aadhar"] = "123456789012"
            session_data["otp_verified"] = True

        response = self.client.post("/verify_face", json={"action": "capture"})
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "verified")
        self.assertTrue(payload["redirect"].endswith("/vote"))

    def test_admin_manage_pages_render_for_authenticated_admin(self):
        with self.client.session_transaction() as session_data:
            session_data["admin_authenticated"] = True
            session_data["admin_username"] = "admin"

        self.assertEqual(self.client.get("/admin/voters").status_code, 200)
        self.assertEqual(self.client.get("/admin/candidates").status_code, 200)
        self.assertEqual(self.client.get("/admin/election-control").status_code, 200)
        self.assertEqual(self.client.get("/admin/analytics").status_code, 200)
        self.assertEqual(self.client.get("/admin/results").status_code, 200)
        self.assertEqual(self.client.get("/admin/logs").status_code, 200)


if __name__ == "__main__":
    unittest.main()
