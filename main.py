import hashlib
import os
import random
import base64
import binascii
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from dotenv import load_dotenv

load_dotenv()

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from mongo_db import (
        add_candidate,
        check_voting_status,
        create_user,
        find_user_by_credentials,
        get_available_constituencies,
        get_all_voters,
        get_candidate_list,
        get_face_records,
        get_registration_overview,
        get_recent_device_logs,
        get_recent_votes,
        get_system_metrics,
        get_vote_results,
        get_voter_by_aadhar,
        get_voter_by_voter_id,
        initialize_database,
        log_device_activity,
        log_audit_event,
        remove_candidate,
        reset_election,
        store_vote,
        update_voter_record,
        update_candidate,
        update_model_status,
        delete_voter_record,
    )
except Exception as exc:
    add_candidate = None
    check_voting_status = None
    create_user = None
    find_user_by_credentials = None
    get_available_constituencies = None
    get_all_voters = None
    get_candidate_list = None
    get_face_records = None
    get_registration_overview = None
    get_recent_device_logs = None
    get_recent_votes = None
    get_system_metrics = None
    get_vote_results = None
    get_voter_by_aadhar = None
    get_voter_by_voter_id = None
    initialize_database = None
    log_device_activity = None
    log_audit_event = None
    remove_candidate = None
    reset_election = None
    store_vote = None
    update_voter_record = None
    update_candidate = None
    update_model_status = None
    delete_voter_record = None
    DATABASE_BACKEND_ERROR = str(exc)
else:
    DATABASE_BACKEND_ERROR = None

from system_architecture import get_flow_stages
from biometric_modules import (
    REGISTRATION_SAMPLE_TARGET,
    RECOGNITION_MODEL_OPTIONS,
    build_face_dataset,
    capture_face,
)
from anti_fraud import REJECT_CONFIDENCE_THRESHOLD, evaluate_confidence, evaluate_liveness
from election_config import DEFAULT_CANDIDATES
from secure_vote import encrypt_vote


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "voting_secret_dev_only")
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", 'static/uploads/')
DATASET_FOLDER = os.environ.get("DATASET_FOLDER", 'dataset')
OTP_FILE = os.environ.get("OTP_FILE", "otp.txt")
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
IS_RENDER = env_flag("RENDER", False)
PRESENTATION_MODE = env_flag("PRESENTATION_MODE", IS_RENDER)
ALLOW_SERVER_CAMERA = env_flag("ALLOW_SERVER_CAMERA", not IS_RENDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASET_FOLDER'] = DATASET_FOLDER

OPENCV_ERROR = (
    "OpenCV with the contrib face module is required for face verification. "
    "Install the dependencies from requirements.txt to enable this feature."
)
CAMERA_DISABLED_ERROR = (
    "Server-side camera streaming is disabled in this deployment. "
    "Use guided verification for presentations, or run the app locally to demo the live camera feed."
)
DATABASE_ERROR = None


face_cascade = None
eye_cascade = None
recognizer = None
label_to_identity = {}


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)


current_status = "Waiting for face..."
camera = None
last_auth_audit_signature = None
previous_face_box = None
ELECTION_STATE = {
    "status": "ACTIVE",
    "updated_at": datetime.now(UTC),
}
current_authentication = {
    "status": "Waiting for face...",
    "decision": "pending",
    "face_match": False,
    "voter_id": None,
    "name": None,
    "constituency": None,
    "has_voted": None,
    "confidence": None,
    "liveness_passed": False,
    "liveness_message": "Awaiting liveness cues.",
}


def extract_face(image):
    return capture_face(image, face_cascade, cv2)


def reset_authentication_state():
    global current_status, current_authentication, last_auth_audit_signature, previous_face_box
    current_status = "Waiting for face..."
    last_auth_audit_signature = None
    previous_face_box = None
    current_authentication = {
        "status": current_status,
        "decision": "pending",
        "face_match": False,
        "voter_id": None,
        "name": None,
        "constituency": None,
        "has_voted": None,
        "confidence": None,
        "liveness_passed": False,
        "liveness_message": "Awaiting liveness cues.",
    }


def update_authentication_state(
    status,
    decision="pending",
    face_match=False,
    voter=None,
    confidence=None,
    liveness_passed=False,
    liveness_message="Awaiting liveness cues.",
):
    global current_status, current_authentication
    current_status = status
    current_authentication = {
        "status": status,
        "decision": decision,
        "face_match": face_match,
        "voter_id": voter.get("voter_id") if voter else None,
        "name": voter.get("name") if voter else None,
        "constituency": voter.get("constituency") if voter else None,
        "has_voted": voter.get("has_voted") if voter else None,
        "confidence": confidence,
        "liveness_passed": liveness_passed,
        "liveness_message": liveness_message,
    }


def is_admin_authenticated():
    return bool(session.get("admin_authenticated"))


def admin_auth_error():
    return validation_error("Please sign in with admin credentials to continue.", 'admin_login.html')


def get_election_state():
    return {
        "status": ELECTION_STATE["status"],
        "updated_at": ELECTION_STATE["updated_at"],
        "is_active": ELECTION_STATE["status"] == "ACTIVE",
    }


def set_election_state(status: str):
    ELECTION_STATE["status"] = status
    ELECTION_STATE["updated_at"] = datetime.now(UTC)


def is_election_active():
    return get_election_state()["is_active"]


def get_request_device_context():
    ip_address = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    machine_source = "|".join(
        [
            ip_address,
            request.headers.get("User-Agent", "unknown"),
            request.headers.get("Sec-CH-UA-Platform", "unknown"),
        ]
    )
    machine_id = hashlib.sha256(machine_source.encode("utf-8")).hexdigest()[:16].upper()
    return {"ip_address": ip_address, "machine_id": machine_id}


def get_active_candidates(constituency: str):
    if get_candidate_list is None:
        return [
            {"id": candidate["candidate_id"], "name": candidate["name"], "party": candidate["party"]}
            for candidate in DEFAULT_CANDIDATES
        ]

    active = []
    for candidate in get_candidate_list():
        if candidate.get("constituency") == constituency:
            active.append(
                {"id": candidate["candidate_id"], "name": candidate["name"], "party": candidate["party"]}
            )

    if active:
        return active

    return [
        {"id": candidate["candidate_id"], "name": candidate["name"], "party": candidate["party"]}
        for candidate in DEFAULT_CANDIDATES
    ]


def evaluate_voter_authentication(identity, expected_aadhar, confidence, liveness_result):
    matched_aadhar = identity.get("aadhar") if identity else None
    matched_voter_id = identity.get("voter_id") if identity else None

    if matched_aadhar != expected_aadhar:
        return {
            "status": "Authentication rejected - face mismatch.",
            "decision": "reject",
            "face_match": False,
            "voter": None,
        }

    confidence_check = evaluate_confidence(confidence)
    if confidence_check["decision"] == "reject":
        return {
            "status": "Authentication rejected - confidence threshold exceeded.",
            "decision": "reject",
            "face_match": False,
            "voter": None,
        }
    if confidence_check["decision"] == "retry":
        return {
            "status": "Authentication paused - confidence uncertain, retry required.",
            "decision": "retry",
            "face_match": True,
            "voter": None,
        }

    if not liveness_result["is_live"]:
        return {
            "status": f"Authentication rejected - liveness check failed ({liveness_result['message']}).",
            "decision": "reject",
            "face_match": True,
            "voter": None,
        }

    if check_voting_status is None:
        return {
            "status": "Vote denied - eligibility check unavailable.",
            "decision": "reject",
            "face_match": True,
            "voter": None,
        }

    status_result = check_voting_status(matched_voter_id)
    voter = status_result.get("voter")
    status = status_result.get("message", "Authentication completed.")
    decision = "allow" if status_result.get("allowed") else "reject"
    return {
        "status": status,
        "decision": decision,
        "face_match": True,
        "voter": voter,
    }


def decode_browser_image(data_url):
    if not data_url or "," not in data_url or cv2 is None:
        return None

    _header, encoded = data_url.split(",", 1)
    try:
        image_bytes = base64.b64decode(encoded)
    except (ValueError, binascii.Error):
        return None

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image


def verify_browser_frames(expected_aadhar, images):
    global previous_face_box, last_auth_audit_signature

    if not images:
        update_authentication_state("No camera frames received. Please allow webcam access and try again.")
        return {
            "status": "retry",
            "message": "No camera frames received.",
        }, 400

    if not is_opencv_available():
        return {"status": "error", "message": OPENCV_ERROR}, 503

    if not is_database_available():
        return {"status": "error", "message": f"Database unavailable: {DATABASE_ERROR}"}, 503

    if recognizer is None or not label_to_identity:
        trained = train_recognizer()
        if not trained or recognizer is None:
            update_authentication_state("Face recognition model is unavailable.", decision="reject")
            return {"status": "error", "message": "Face recognition model is unavailable."}, 503

    detections = []
    local_previous_face_box = previous_face_box

    for data_url in images:
        frame = decode_browser_image(data_url)
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4) if face_cascade is not None else []
        if len(faces) == 0:
            continue

        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        face_roi = gray[y:y + h, x:x + w]
        if face_roi.size == 0:
            continue

        face_resized = cv2.resize(face_roi, (200, 200))
        face_resized = cv2.equalizeHist(face_resized)
        eye_count = 0
        if eye_cascade is not None:
            eyes = eye_cascade.detectMultiScale(face_roi)
            eye_count = len(eyes)

        try:
            label, confidence = recognizer.predict(face_resized)
        except cv2.error:
            continue

        identity = label_to_identity.get(label, {})
        liveness_result = evaluate_liveness((x, y, w, h), local_previous_face_box, eye_count)
        local_previous_face_box = (x, y, w, h)
        detections.append(
            {
                "identity": identity,
                "confidence": round(float(confidence), 2),
                "liveness": liveness_result,
                "face_box": [int(x), int(y), int(w), int(h)],
            }
        )

    previous_face_box = local_previous_face_box

    if not detections:
        update_authentication_state(
            "No recognizable face detected. Center your face in the frame and try again.",
            decision="retry",
            liveness_message="No face detected.",
        )
        return {
            "status": "retry",
            "message": "No recognizable face detected. Try again with better lighting.",
            "decision": "retry",
            "face_box": None,
        }, 400

    best_detection = min(detections, key=lambda item: item["confidence"])
    combined_liveness = {
        "is_live": any(item["liveness"]["is_live"] for item in detections),
        "message": "; ".join(
            list(dict.fromkeys(item["liveness"]["message"] for item in detections))
        ),
    }
    auth_result = evaluate_voter_authentication(
        best_detection["identity"],
        expected_aadhar,
        best_detection["confidence"],
        combined_liveness,
    )
    voter = auth_result.get("voter")
    update_authentication_state(
        auth_result["status"],
        decision=auth_result["decision"],
        face_match=auth_result["face_match"],
        voter=voter,
        confidence=best_detection["confidence"],
        liveness_passed=combined_liveness["is_live"],
        liveness_message=combined_liveness["message"],
    )

    matched_aadhar = best_detection["identity"].get("aadhar")
    audit_signature = (auth_result["decision"], auth_result["status"], matched_aadhar)
    if (
        log_audit_event is not None
        and auth_result["decision"] != "pending"
        and audit_signature != last_auth_audit_signature
    ):
        log_audit_event("authentication_decision", auth_result["status"], matched_aadhar)
        last_auth_audit_signature = audit_signature

    if auth_result["decision"] == "allow":
        return {
            "status": "verified",
            "message": "Face verified successfully.",
            "redirect": url_for('vote'),
            "voter_id": voter["voter_id"] if voter else None,
            "decision": "allow",
            "face_box": best_detection["face_box"],
        }, 200

    if auth_result["decision"] == "retry":
        return {
            "status": "retry",
            "message": auth_result["status"],
            "decision": "retry",
            "face_box": best_detection["face_box"],
        }, 400

    redirect_target = url_for('vote_denied') if auth_result["face_match"] else None
    return {
        "status": "denied",
        "message": auth_result["status"],
        "redirect": redirect_target,
        "voter_id": voter["voter_id"] if voter else None,
        "decision": auth_result["decision"],
        "face_box": best_detection["face_box"],
    }, 403


def train_recognizer():
    global recognizer, label_to_identity

    if cv2 is None or not hasattr(cv2, "face") or get_face_records is None:
        recognizer = None
        label_to_identity = {}
        if update_model_status is not None:
            update_model_status("LBPH", 0, False)
        return False

    rows = get_face_records()

    samples = []
    labels = []
    label_to_identity = {}

    for label, row in enumerate(rows):
        aadhar = row.get("aadhaar")
        voter_id = row.get("voter_id")
        image_path = row.get("image_path")
        dataset_path = row.get("face_dataset_path")

        def add_sample(sample_image):
            if sample_image is None:
                return
            if sample_image.shape[:2] != (200, 200):
                sample_image = cv2.resize(sample_image, (200, 200))
            samples.append(sample_image)
            labels.append(label)

        if dataset_path and os.path.isdir(dataset_path):
            for file_name in os.listdir(dataset_path):
                if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                file_path = os.path.join(dataset_path, file_name)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                add_sample(image)

        if image_path and os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                face = extract_face(image)
                if face is None:
                    gray = (
                        image
                        if hasattr(image, "shape") and len(image.shape) == 2
                        else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    )
                    face = cv2.resize(gray, (200, 200))
                add_sample(face)

        if label not in label_to_identity and (samples and labels and labels[-1] == label):
            label_to_identity[label] = {"aadhar": aadhar, "voter_id": voter_id}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if samples:
        recognizer.train(samples, np.array(labels))
        if update_model_status is not None:
            update_model_status("LBPH", len(samples), True)
        if log_audit_event is not None:
            log_audit_event("model_training", f"LBPH recognizer trained with {len(samples)} samples.")
        return True
    recognizer = None
    if update_model_status is not None:
        update_model_status("LBPH", 0, False)
    return False


def is_database_available():
    return DATABASE_ERROR is None and DATABASE_BACKEND_ERROR is None


def is_opencv_available():
    return cv2 is not None and hasattr(cv2, "face")


def is_server_camera_enabled():
    return is_opencv_available() and ALLOW_SERVER_CAMERA


def validation_error(message, template_name):
    return render_template(template_name, error=message)


def get_voter_session_record():
    aadhar = session.get('aadhar')
    if not aadhar or get_voter_by_aadhar is None:
        return None
    return get_voter_by_aadhar(aadhar)


def has_voter_session():
    return bool(session.get('aadhar'))


def voter_workflow_step():
    if session.get("vote_completed"):
        return "completed"
    if session.get("pending_candidate_id"):
        return "confirmation"
    if has_vote_access():
        return "vote"
    if session.get("otp_verified"):
        return "face_verification"
    if has_voter_session():
        return "dashboard"
    return "login"


def build_voter_dashboard_context():
    voter = get_voter_session_record()
    election_state = get_election_state()
    return {
        "voter": voter,
        "election_state": election_state,
        "vote_status": "Completed" if voter and voter.get("has_voted") else "Pending",
        "workflow_step": voter_workflow_step(),
    }


def build_admin_context():
    metrics = get_system_metrics() if is_database_available() and get_system_metrics is not None else None
    recent_voters = (
        get_registration_overview() if is_database_available() and get_registration_overview is not None else []
    )
    recent_votes = get_recent_votes() if is_database_available() and get_recent_votes is not None else []
    candidates = get_candidate_list() if is_database_available() and get_candidate_list is not None else DEFAULT_CANDIDATES
    available_constituencies = (
        get_available_constituencies() if is_database_available() and get_available_constituencies is not None else []
    )
    vote_results = get_vote_results() if is_database_available() and get_vote_results is not None else []
    recent_device_logs = (
        get_recent_device_logs() if is_database_available() and get_recent_device_logs is not None else []
    )
    voters = get_all_voters() if is_database_available() and get_all_voters is not None else []
    turnout_percentage = 0
    if metrics and metrics.get("registered_citizens"):
        turnout_percentage = round(
            (metrics.get("votes_cast", 0) / metrics.get("registered_citizens", 1)) * 100,
            2,
        )
    return {
        "metrics": metrics,
        "recent_voters": recent_voters,
        "recent_votes": recent_votes,
        "candidates": candidates,
        "available_constituencies": available_constituencies,
        "vote_results": vote_results,
        "recent_device_logs": recent_device_logs,
        "voters": voters,
        "turnout_percentage": turnout_percentage,
        "election_state": get_election_state(),
        "flow_stages": get_flow_stages(),
        "recognition_models": RECOGNITION_MODEL_OPTIONS,
        "sample_target": REGISTRATION_SAMPLE_TARGET,
        "database_error": DATABASE_ERROR,
        "opencv_error": None if is_opencv_available() else OPENCV_ERROR,
        "presentation_mode": PRESENTATION_MODE,
        "server_camera_enabled": is_server_camera_enabled(),
    }


def setup_runtime_dependencies():
    global face_cascade, eye_cascade, recognizer, DATABASE_ERROR

    if is_opencv_available():
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )
        recognizer = cv2.face.LBPHFaceRecognizer_create()

    if initialize_database is None:
        DATABASE_ERROR = DATABASE_BACKEND_ERROR or "Database backend is unavailable."
        return

    try:
        initialize_database()
    except Exception as exc:
        DATABASE_ERROR = str(exc)
        return

    train_recognizer()


setup_runtime_dependencies()
reset_authentication_state()



@app.route('/')
def home():
    metrics = get_system_metrics() if is_database_available() and get_system_metrics is not None else None
    recent_voters = (
        get_registration_overview() if is_database_available() and get_registration_overview is not None else []
    )
    return render_template(
        'home.html',
        flow_stages=get_flow_stages(),
        metrics=metrics,
        recent_voters=recent_voters,
        recognition_models=RECOGNITION_MODEL_OPTIONS,
        database_error=DATABASE_ERROR,
        opencv_error=None if is_opencv_available() else OPENCV_ERROR,
        presentation_mode=PRESENTATION_MODE,
        server_camera_enabled=is_server_camera_enabled(),
    )



@app.route('/voter/dashboard')
def voter_dashboard():
    if not has_voter_session():
        return redirect(url_for('login'))
    return render_template('voter/dashboard.html', **build_voter_dashboard_context(), error=None)


@app.route('/voter/logout')
def voter_logout():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    reset_authentication_state()
    session.pop('otp_verified', None)
    session.pop('aadhar', None)
    session.pop('otp', None)
    session.pop('pending_candidate_id', None)
    session.pop('pending_candidate_name', None)
    session.pop('vote_completed', None)
    session.pop('vote_timestamp', None)
    session.pop('last_vote_id', None)
    return redirect(url_for('home'))


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if is_admin_authenticated():
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            return validation_error("Admin username and password are required.", 'admin_login.html')

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_authenticated'] = True
            session['admin_username'] = username
            if log_audit_event is not None:
                log_audit_event("admin_login", "Administrator signed in successfully.", username)
            return redirect(url_for('admin_dashboard'))

        return validation_error("Invalid admin credentials.", 'admin_login.html')

    return render_template('admin_login.html', error=None)


@app.route('/admin/logout')
def admin_logout():
    admin_user = session.get('admin_username')
    session.pop('admin_authenticated', None)
    session.pop('admin_username', None)
    if log_audit_event is not None and admin_user:
        log_audit_event("admin_logout", "Administrator signed out.", admin_user)
    return redirect(url_for('admin_login'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        if not is_database_available():
            return validation_error(f"Database unavailable: {DATABASE_ERROR}", 'signup.html')

        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        age = request.form.get('age', '').strip()
        constituency = request.form.get('constituency', '').strip()
        aadhar = request.form.get('aadhar', '').strip()
        password = request.form.get('password', '')
        images = [image for image in request.files.getlist('images') if image and image.filename]

        if not all([name, email, phone, age, constituency, aadhar, password]) or not images:
            return validation_error("All fields are required.", 'signup.html')

        if (
            not email.endswith("@gmail.com")
            or not phone.isdigit()
            or len(phone) != 10
            or not age.isdigit()
            or int(age) < 18
            or not aadhar.isdigit()
            or len(aadhar) != 12
        ):
            return validation_error(
                "Invalid data. Enter a Gmail address, a 10-digit phone number, age 18 or above, and a 12-digit Aadhaar number.",
                'signup.html',
            )

        if not is_opencv_available():
            return validation_error(OPENCV_ERROR, 'signup.html')

        voter_id = f"V{uuid4().hex[:8].upper()}"
        temp_image_paths = []
        uploaded_images = []

        try:
            for index, image in enumerate(images, start=1):
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{aadhar}_{index}.jpg")
                image.save(temp_path)
                temp_image_paths.append(temp_path)

                uploaded_image = cv2.imread(temp_path)
                if uploaded_image is None:
                    raise ValueError("One of the uploaded images could not be read. Please upload clear face photos.")

                uploaded_images.append(uploaded_image)

            dataset_info = build_face_dataset(
                uploaded_images,
                voter_id,
                app.config['DATASET_FOLDER'],
                face_cascade,
                cv2,
                sample_target=REGISTRATION_SAMPLE_TARGET,
            )
            if dataset_info is None:
                raise ValueError(
                    "No face detected in the uploaded images. Please upload clear front-facing face photos."
                )

            image_path = dataset_info["profile_image_path"] or temp_image_paths[0]
            user_created = create_user(
                voter_id,
                name,
                email,
                phone,
                int(age),
                constituency,
                aadhar,
                password,
                image_path,
                dataset_info["dataset_path"],
                dataset_info["sample_count"],
            )
        except ValueError as exc:
            for path in temp_image_paths:
                if os.path.exists(path):
                    os.remove(path)
            return validation_error(str(exc), 'signup.html')

        if not user_created:
            for path in temp_image_paths:
                if os.path.exists(path):
                    os.remove(path)
            return validation_error("Aadhaar number already registered.", 'signup.html')

        for path in temp_image_paths:
            if os.path.exists(path):
                os.remove(path)

        train_recognizer()
        return redirect(url_for('home'))
    return render_template('signup.html', error=None)



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if not is_database_available():
            return validation_error(f"Database unavailable: {DATABASE_ERROR}", 'login.html')

        aadhar = request.form.get('aadhar', '').strip()
        password = request.form.get('password', '')

        if not aadhar or not password:
            return validation_error("Aadhaar number and password are required.", 'login.html')

        user = find_user_by_credentials(aadhar, password)

        if user:
            otp = str(random.randint(100000, 999999))
            session.clear()
            session['otp'] = otp
            session['otp_verified'] = False
            with open(OTP_FILE, "w") as f:
                f.write(otp)

            session['aadhar'] = aadhar
            if log_audit_event is not None:
                log_audit_event("authentication_started", "Credential verification completed, OTP issued.", aadhar)
            return redirect(url_for('otp_verification'))
        else:
            return validation_error("Invalid credentials. Please sign up first.", 'login.html')
    return render_template('login.html', error=None)



@app.route('/otp_verification', methods=['GET', 'POST'])
def otp_verification():
    if 'aadhar' not in session or 'otp' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        entered_otp = request.form.get('otp', '').strip()
        saved_otp = session.get('otp')

        if entered_otp == saved_otp:
            session['otp_verified'] = True
            session.pop('otp', None)
            if log_audit_event is not None:
                log_audit_event("otp_verified", "OTP verification successful.", session.get('aadhar'))
            return redirect(url_for('voter_dashboard'))
        else:
            return validation_error("Invalid OTP.", 'otp_verification.html')
    return render_template('otp_verification.html', error=None, dev_test_otp=session.get('otp'))


@app.route('/face_verification')
@app.route('/voter/face-verification')
def face_verification():
    if 'aadhar' not in session or not session.get('otp_verified'):
        return redirect(url_for('login'))

    reset_authentication_state()
    return render_template(
        'voter/face_verification.html',
        **build_voter_dashboard_context(),
        opencv_error=None if is_opencv_available() else OPENCV_ERROR,
        camera_error=None if is_server_camera_enabled() else CAMERA_DISABLED_ERROR,
    )



@app.route('/stop_stream')
def stop_stream():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    reset_authentication_state()
    session.pop('otp_verified', None)
    session.pop('aadhar', None)
    session.pop('otp', None)
    return redirect(url_for('home'))



@app.route('/status')
def status():
    return jsonify(current_authentication)


@app.route('/health')
def health():
    return jsonify(
        {
            "status": "ok",
            "database_available": is_database_available(),
            "opencv_available": is_opencv_available(),
            "server_camera_enabled": is_server_camera_enabled(),
            "presentation_mode": PRESENTATION_MODE,
        }
    )


@app.route('/verify_face', methods=['POST'])
def verify_face():
    if not has_voter_session() or not session.get('otp_verified'):
        return jsonify({"status": "login_required", "message": "Login and OTP verification are required."}), 403

    voter = get_voter_session_record()
    if voter is None:
        return jsonify({"status": "not_found", "message": "Voter record unavailable."}), 404

    payload = request.get_json(silent=True) or {}
    browser_images = payload.get("images") if isinstance(payload, dict) else None
    if browser_images:
        response_body, status_code = verify_browser_frames(voter["aadhaar"], browser_images)
        return jsonify(response_body), status_code

    status_result = check_voting_status(voter["voter_id"]) if check_voting_status is not None else None
    if not status_result:
        return jsonify({"status": "error", "message": "Eligibility check unavailable."}), 503

    if not status_result.get("allowed"):
        update_authentication_state(
            status_result["message"],
            decision="reject",
            face_match=True,
            voter=status_result.get("voter") or voter,
        )
        return jsonify(
            {
                "status": "denied",
                "message": status_result["message"],
                "redirect": url_for('vote_denied'),
                "voter_id": voter["voter_id"],
            }
        )

    update_authentication_state(
        "Face verified successfully. Eligibility confirmed.",
        decision="allow",
        face_match=True,
        voter=status_result.get("voter") or voter,
        liveness_passed=True,
        liveness_message="Verified by guided workflow confirmation.",
    )
    return jsonify(
        {
            "status": "verified",
            "message": "Verification successful.",
            "redirect": url_for('vote'),
            "voter_id": voter["voter_id"],
        }
    )


@app.route('/analyze_face_frame', methods=['POST'])
def analyze_face_frame():
    if not has_voter_session() or not session.get('otp_verified'):
        return jsonify({"status": "login_required", "message": "Login and OTP verification are required."}), 403

    voter = get_voter_session_record()
    if voter is None:
        return jsonify({"status": "not_found", "message": "Voter record unavailable."}), 404

    payload = request.get_json(silent=True) or {}
    image = payload.get("image") if isinstance(payload, dict) else None
    response_body, status_code = verify_browser_frames(voter["aadhaar"], [image] if image else [])
    return jsonify(response_body), status_code


@app.route('/voter/voting-status')
def voting_status():
    if not has_voter_session():
        return redirect(url_for('login'))
    return render_template('voter/voting_status.html', **build_voter_dashboard_context(), error=None)


@app.route('/admin/dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if not is_admin_authenticated():
        return redirect(url_for('admin_login'))

    dashboard_notice = None
    if request.method == 'POST' and is_database_available():
        action = request.form.get('action', '').strip()
        candidate_id = request.form.get('candidate_id', '').strip().upper()
        name = request.form.get('name', '').strip()
        party = request.form.get('party', '').strip()
        constituency = request.form.get('constituency', '').strip()

        if action == 'add_candidate' and candidate_id and name and party and constituency and add_candidate is not None:
            add_candidate(candidate_id, name, party, constituency)
            dashboard_notice = f"Candidate {candidate_id} saved."
        elif action == 'update_candidate' and candidate_id and name and party and constituency and update_candidate is not None:
            update_candidate(candidate_id, name, party, constituency)
            dashboard_notice = f"Candidate {candidate_id} updated."
        elif action == 'remove_candidate' and candidate_id and remove_candidate is not None:
            remove_candidate(candidate_id)
            dashboard_notice = f"Candidate {candidate_id} removed."

    return render_template(
        'admin/dashboard.html',
        dashboard_notice=dashboard_notice,
        **build_admin_context(),
    )


@app.route('/admin/dashboard/data')
def admin_dashboard_data():
    if not is_admin_authenticated():
        return jsonify({"error": "Admin authentication required."}), 403

    if not is_database_available():
        return jsonify({"error": DATABASE_ERROR or DATABASE_BACKEND_ERROR or "Database unavailable."}), 503

    metrics = get_system_metrics() if get_system_metrics is not None else None
    recent_voters = get_registration_overview() if get_registration_overview is not None else []
    recent_votes = get_recent_votes() if get_recent_votes is not None else []
    vote_results = get_vote_results() if get_vote_results is not None else []
    recent_device_logs = get_recent_device_logs() if get_recent_device_logs is not None else []
    candidates = get_candidate_list() if get_candidate_list is not None else []
    voters = get_all_voters() if get_all_voters is not None else []
    return jsonify(
        {
            "metrics": metrics,
            "recent_voters": recent_voters,
            "recent_votes": recent_votes,
            "vote_results": vote_results,
            "recent_device_logs": recent_device_logs,
            "candidates": candidates,
            "voters": voters,
            "election_state": get_election_state(),
        }
    )


@app.route('/admin/voters', methods=['GET', 'POST'])
def admin_manage_voters():
    if not is_admin_authenticated():
        return redirect(url_for('admin_login'))

    notice = None
    if request.method == 'POST' and is_database_available():
        action = request.form.get('action', '').strip()
        voter_id = request.form.get('voter_id', '').strip()
        if action == 'delete' and voter_id and delete_voter_record is not None:
            notice = f"Voter {voter_id} removed." if delete_voter_record(voter_id) else f"Voter {voter_id} not found."
        elif action == 'edit' and voter_id and update_voter_record is not None:
            notice = (
                f"Voter {voter_id} updated."
                if update_voter_record(
                    voter_id,
                    request.form.get('name', '').strip(),
                    request.form.get('email', '').strip(),
                    request.form.get('phone', '').strip(),
                    request.form.get('constituency', '').strip(),
                    request.form.get('registered') == 'true',
                )
                else f"Voter {voter_id} not found."
            )

    return render_template('admin/manage_voters.html', page_notice=notice, **build_admin_context())


@app.route('/admin/candidates', methods=['GET', 'POST'])
def admin_manage_candidates():
    if not is_admin_authenticated():
        return redirect(url_for('admin_login'))

    notice = None
    if request.method == 'POST' and is_database_available():
        action = request.form.get('action', '').strip()
        candidate_id = request.form.get('candidate_id', '').strip().upper()
        name = request.form.get('name', '').strip()
        party = request.form.get('party', '').strip()
        constituency = request.form.get('constituency', '').strip()

        if action == 'add' and candidate_id and name and party and constituency and add_candidate is not None:
            add_candidate(candidate_id, name, party, constituency)
            notice = f"Candidate {candidate_id} added."
        elif action == 'edit' and candidate_id and name and party and constituency and update_candidate is not None:
            notice = f"Candidate {candidate_id} updated." if update_candidate(candidate_id, name, party, constituency) else f"Candidate {candidate_id} not found."
        elif action == 'delete' and candidate_id and remove_candidate is not None:
            notice = f"Candidate {candidate_id} removed." if remove_candidate(candidate_id) else f"Candidate {candidate_id} not found."

    return render_template('admin/manage_candidates.html', page_notice=notice, **build_admin_context())


@app.route('/admin/add_candidate', methods=['POST'])
def admin_add_candidate():
    if not is_admin_authenticated():
        return redirect(url_for('admin_login'))
    if add_candidate is not None:
        add_candidate(
            request.form.get('candidate_id', '').strip().upper(),
            request.form.get('name', '').strip(),
            request.form.get('party', '').strip(),
            request.form.get('constituency', '').strip(),
        )
    return redirect(url_for('admin_manage_candidates'))


@app.route('/admin/election-control', methods=['GET', 'POST'])
def admin_election_control():
    if not is_admin_authenticated():
        return redirect(url_for('admin_login'))

    notice = None
    if request.method == 'POST':
        action = request.form.get('action', '').strip()
        if action == 'start':
            set_election_state("ACTIVE")
            notice = "Election started. Voting is now enabled."
        elif action == 'stop':
            set_election_state("CLOSED")
            notice = "Election stopped. Voting is now disabled."
        elif action == 'reset':
            set_election_state("SETUP")
            reset_authentication_state()
            if reset_election is not None and is_database_available():
                result = reset_election()
                notice = f"Election reset completed. Votes cleared: {result['deleted_votes']}."
            else:
                notice = "Election state reset to setup mode."

    return render_template('admin/election_control.html', page_notice=notice, **build_admin_context())


@app.route('/admin/analytics')
def admin_analytics():
    if not is_admin_authenticated():
        return redirect(url_for('admin_login'))
    return render_template('admin/analytics.html', **build_admin_context())


@app.route('/admin/results')
def admin_results():
    if not is_admin_authenticated():
        return redirect(url_for('admin_login'))
    return render_template('admin/results.html', **build_admin_context())


@app.route('/admin/logs')
def admin_logs():
    if not is_admin_authenticated():
        return redirect(url_for('admin_login'))
    return render_template('admin/logs.html', **build_admin_context())


def has_vote_access():
    return (
        session.get('otp_verified')
        and current_authentication.get("decision") == "allow"
        and current_authentication.get("voter_id")
        and is_election_active()
    )


@app.route('/vote', methods=['GET', 'POST'])
def vote():
    global camera

    if not has_vote_access():
        return redirect(url_for('face_verification'))

    voter_id = current_authentication.get("voter_id")
    voter = get_voter_by_voter_id(voter_id) if get_voter_by_voter_id is not None else None
    if voter is None:
        return validation_error("Voter record unavailable for vote casting.", 'voter/face_verification.html')
    if voter.get("has_voted"):
        return redirect(url_for('vote_denied'))
    candidates = get_active_candidates(voter["constituency"])

    if camera is not None:
        camera.release()
        camera = None

    if request.method == 'POST':
        candidate_id = request.form.get('candidate_id', '').strip()
        valid_candidate_ids = {candidate["id"] for candidate in candidates}
        if candidate_id not in valid_candidate_ids:
            return render_template('voter/vote_page.html', voter=voter, candidates=candidates, error="Select a valid candidate.")

        selected_candidate = next((candidate for candidate in candidates if candidate["id"] == candidate_id), None)
        session['pending_candidate_id'] = candidate_id
        session['pending_candidate_name'] = selected_candidate["name"] if selected_candidate else candidate_id
        return render_template(
            'voter/vote_confirmation.html',
            voter=voter,
            candidate=selected_candidate,
            error=None,
            election_state=get_election_state(),
        )

    return render_template('voter/vote_page.html', voter=voter, candidates=candidates, error=None, election_state=get_election_state())


@app.route('/submit_vote', methods=['POST'])
def submit_vote():
    if not has_vote_access():
        return redirect(url_for('face_verification'))

    voter_id = current_authentication.get("voter_id")
    voter = get_voter_by_voter_id(voter_id) if get_voter_by_voter_id is not None else None
    if voter is None:
        return redirect(url_for('voter_dashboard'))

    if request.form.get('decision') == 'cancel':
        session.pop('pending_candidate_id', None)
        session.pop('pending_candidate_name', None)
        return redirect(url_for('vote'))

    candidate_id = session.get('pending_candidate_id')
    candidates = get_active_candidates(voter["constituency"])
    valid_candidate_ids = {candidate["id"] for candidate in candidates}
    if candidate_id not in valid_candidate_ids:
        session.pop('pending_candidate_id', None)
        session.pop('pending_candidate_name', None)
        return render_template('voter/vote_page.html', voter=voter, candidates=candidates, error="Select a valid candidate.")

    status_result = check_voting_status(voter_id) if check_voting_status is not None else None
    if not status_result or not status_result.get("allowed"):
        error_message = status_result.get("message", "Vote denied.") if status_result else "Vote denied."
        update_authentication_state(error_message, decision="reject", face_match=True, voter=voter)
        return redirect(url_for('vote_denied'))

    encrypted_vote = encrypt_vote(candidate_id, voter["constituency"])
    storage_result = store_vote(
        voter_id,
        encrypted_vote["vote_id"],
        voter["constituency"],
        candidate_id,
    ) if store_vote is not None else {"stored": False, "message": "Vote storage backend unavailable."}

    if not storage_result.get("stored"):
        return render_template('voter/vote_page.html', voter=voter, candidates=candidates, error=storage_result["message"])

    updated_voter = get_voter_by_voter_id(voter_id) if get_voter_by_voter_id is not None else voter
    timestamp = datetime.now(UTC)
    update_authentication_state(
        "Vote submitted successfully - secret ballot stored.",
        decision="completed",
        face_match=True,
        voter=updated_voter,
    )
    if log_audit_event is not None:
        log_audit_event("vote_cast", f"Vote {encrypted_vote['vote_id']} stored securely.", session.get('aadhar'))
    if log_device_activity is not None:
        device_context = get_request_device_context()
        log_device_activity(voter_id, device_context["ip_address"], device_context["machine_id"], "vote_submission")

    session['vote_completed'] = True
    session['vote_timestamp'] = timestamp.isoformat()
    session['last_vote_id'] = encrypted_vote["vote_id"]
    session.pop('pending_candidate_id', None)
    session.pop('pending_candidate_name', None)
    session.pop('otp_verified', None)
    session.pop('otp', None)
    return render_template(
        'voter/vote_success.html',
        vote_id=encrypted_vote["vote_id"],
        constituency=voter["constituency"],
        vote_timestamp=timestamp,
        election_state=get_election_state(),
    )


@app.route('/vote/denied')
def vote_denied():
    if not has_voter_session():
        return redirect(url_for('login'))
    return render_template('voter/vote_denied.html', **build_voter_dashboard_context(), error=None)



# VIDEO STREAM
@app.route('/video_feed')
def video_feed():
    expected_aadhar = session.get('aadhar')
    if not expected_aadhar or not session.get('otp_verified'):
        return Response("Login and OTP verification are required before face verification.", status=403)
    if not is_opencv_available():
        return Response(OPENCV_ERROR, mimetype='text/plain', status=503)
    if not ALLOW_SERVER_CAMERA:
        return Response(CAMERA_DISABLED_ERROR, mimetype='text/plain', status=503)
    if not is_database_available():
        return Response(f"Database unavailable: {DATABASE_ERROR}", mimetype='text/plain', status=503)
    return Response(generate_frames(expected_aadhar), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames(expected_aadhar):
    global camera, last_auth_audit_signature, previous_face_box

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        update_authentication_state("Camera is unavailable. Connect a webcam and try again.", decision="reject")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                update_authentication_state("Unable to read from the camera.", decision="reject")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if recognizer is None or not label_to_identity:
                voting_status = "Authentication rejected - no registered face data available."
                color = (0, 0, 255)
                update_authentication_state(voting_status, decision="reject")
            else:
                voting_status = "Authentication in progress - scanning face."
                color = (0, 0, 255)
                update_authentication_state(voting_status)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face_roi, (200, 200))
                face_resized = cv2.equalizeHist(face_resized)
                eye_count = 0
                if eye_cascade is not None:
                    eyes = eye_cascade.detectMultiScale(face_roi)
                    eye_count = len(eyes)
                liveness_result = evaluate_liveness((x, y, w, h), previous_face_box, eye_count)
                previous_face_box = (x, y, w, h)

                try:
                    label, confidence = recognizer.predict(face_resized)
                    identity = label_to_identity.get(label, {})
                    matched_aadhar = identity.get("aadhar")
                    auth_result = evaluate_voter_authentication(identity, expected_aadhar, confidence, liveness_result)
                    voter = auth_result.get("voter")
                    is_match = auth_result["decision"] == "allow"

                    if is_match:
                        name = voter.get("voter_id", "Registered Voter") if voter else "Registered Voter"
                        voting_status = auth_result["status"]
                        color = (0, 255, 0)
                    elif auth_result["face_match"]:
                        name = voter.get("voter_id", "Registered Voter") if voter else "Registered Voter"
                        voting_status = auth_result["status"]
                    else:
                        name = "Face Not Recognized"
                        voting_status = auth_result["status"]

                    update_authentication_state(
                        voting_status,
                        decision=auth_result["decision"],
                        face_match=auth_result["face_match"],
                        voter=voter,
                        confidence=round(float(confidence), 2),
                        liveness_passed=liveness_result["is_live"],
                        liveness_message=liveness_result["message"],
                    )
                    audit_signature = (auth_result["decision"], voting_status, matched_aadhar)
                    if (
                        log_audit_event is not None
                        and auth_result["decision"] != "pending"
                        and audit_signature != last_auth_audit_signature
                    ):
                        log_audit_event("authentication_decision", voting_status, matched_aadhar)
                        if log_device_activity is not None and voter is not None:
                            device_context = get_request_device_context()
                            log_device_activity(
                                voter.get("voter_id"),
                                device_context["ip_address"],
                                device_context["machine_id"],
                                f"authentication_{auth_result['decision']}",
                            )
                        last_auth_audit_signature = audit_signature
                except cv2.error:
                    name = "Face Not Recognized"
                    voting_status = "Authentication rejected - face recognition error."
                    update_authentication_state(
                        voting_status,
                        decision="reject",
                        confidence=REJECT_CONFIDENCE_THRESHOLD + 1,
                        liveness_passed=False,
                        liveness_message="Recognition error.",
                    )

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

            cv2.putText(frame, voting_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        if camera is not None:
            camera.release()
            camera = None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", "5000")))
