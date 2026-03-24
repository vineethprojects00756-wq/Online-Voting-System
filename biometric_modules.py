import os
from typing import Dict, List, Optional


REGISTRATION_SAMPLE_TARGET = 30


def ensure_directory(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def detect_face(image, face_cascade, cv2_module):
    if image is None or face_cascade is None or cv2_module is None:
        return None

    if hasattr(image, "shape") and len(image.shape) == 2:
        gray = image
    else:
        gray = cv2_module.cvtColor(image, cv2_module.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    return gray, (x, y, w, h)


def crop_face(gray_image, bounds):
    if gray_image is None or bounds is None:
        return None

    x, y, w, h = bounds
    return gray_image[y:y + h, x:x + w]


def convert_to_grayscale(face_image, cv2_module, size=(200, 200)):
    if face_image is None or cv2_module is None:
        return None
    resized = cv2_module.resize(face_image, size)
    # Normalize lighting to reduce mismatches between registration and live capture.
    return cv2_module.equalizeHist(resized)


def capture_face(image, face_cascade, cv2_module):
    detection_result = detect_face(image, face_cascade, cv2_module)
    if detection_result is None:
        return None

    gray_image, bounds = detection_result
    cropped_face = crop_face(gray_image, bounds)
    return convert_to_grayscale(cropped_face, cv2_module)


def store_dataset(samples: List, voter_dir: str, cv2_module) -> str:
    ensure_directory(voter_dir)
    for index, sample in enumerate(samples, start=1):
        sample_path = os.path.join(voter_dir, f"sample_{index:02d}.jpg")
        cv2_module.imwrite(sample_path, sample)
    return voter_dir


def _rotate_face(face_image, cv2_module, angle: int):
    if angle == 0:
        return face_image

    height, width = face_image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2_module.getRotationMatrix2D(center, angle, 1.0)
    return cv2_module.warpAffine(face_image, matrix, (width, height))


def _shift_brightness(face_image, cv2_module, beta: int):
    return cv2_module.convertScaleAbs(face_image, alpha=1.0, beta=beta)


def build_face_dataset(
    images: List,
    voter_id: str,
    dataset_dir: str,
    face_cascade,
    cv2_module,
    sample_target: int = REGISTRATION_SAMPLE_TARGET,
) -> Optional[Dict[str, object]]:
    if not images or face_cascade is None or cv2_module is None:
        return None

    voter_dir = ensure_directory(os.path.join(dataset_dir, voter_id))
    variations = [
        lambda face: face,
        lambda face: cv2_module.flip(face, 1),
        lambda face: _rotate_face(face, cv2_module, -7),
        lambda face: _rotate_face(face, cv2_module, 7),
        lambda face: _shift_brightness(face, cv2_module, -18),
        lambda face: _shift_brightness(face, cv2_module, 18),
    ]

    samples = []
    profile_image_path = None

    for image in images:
        face = capture_face(image, face_cascade, cv2_module)
        if face is None:
            continue

        if profile_image_path is None:
            profile_image_path = os.path.join(voter_dir, "profile.jpg")
            cv2_module.imwrite(profile_image_path, face)

        for variation in variations:
            sample = variation(face)
            if sample is None:
                continue

            samples.append(sample)
            if len(samples) >= sample_target:
                break

        if len(samples) >= sample_target:
            break

    if not samples:
        return None

    while len(samples) < sample_target:
        base_sample = samples[len(samples) % len(samples)]
        samples.append(cv2_module.flip(base_sample, 1))

    store_dataset(samples, voter_dir, cv2_module)

    return {
        "dataset_path": voter_dir,
        "profile_image_path": profile_image_path,
        "sample_count": len(samples),
    }


RECOGNITION_MODEL_OPTIONS = [
    {
        "name": "LBPH Face Recognizer",
        "status": "active",
        "tag": "Current Project",
        "summary": "Fast CPU-friendly recognizer suited to small and medium voter datasets.",
        "workflow": [
            "Dataset images",
            "Feature extraction",
            "LBPH training",
            "face_model.yml",
        ],
        "advantages": ["Fast", "Works on CPU", "Good for small datasets"],
    },
    {
        "name": "FaceNet / Dlib Embeddings",
        "status": "planned",
        "tag": "Advanced AI",
        "summary": "Deep-learning face embeddings with similarity matching for higher-accuracy verification.",
        "workflow": [
            "Image",
            "Face detection",
            "128D embedding",
            "Cosine similarity comparison",
        ],
        "advantages": ["Better accuracy", "Embedding-based verification", "More robust matching"],
    },
]
