import math


ACCEPT_CONFIDENCE_THRESHOLD = 60
REJECT_CONFIDENCE_THRESHOLD = 95


def evaluate_confidence(confidence: float) -> dict:
    if confidence < ACCEPT_CONFIDENCE_THRESHOLD:
        return {"decision": "accept", "message": "Face confidence accepted."}
    if confidence > REJECT_CONFIDENCE_THRESHOLD:
        return {"decision": "reject", "message": "Face confidence rejected."}
    return {"decision": "retry", "message": "Face confidence uncertain. Retry required."}


def evaluate_liveness(face_box, previous_face_box, eye_count: int) -> dict:
    blink_detected = eye_count == 0
    head_movement_detected = False
    depth_variation_detected = False

    if previous_face_box and face_box:
        prev_x, prev_y, prev_w, prev_h = previous_face_box
        x, y, w, h = face_box
        center_shift = math.hypot((x + w / 2) - (prev_x + prev_w / 2), (y + h / 2) - (prev_y + prev_h / 2))
        head_movement_detected = center_shift > 12
        depth_variation_detected = abs((w * h) - (prev_w * prev_h)) > 1800

    is_live = blink_detected or head_movement_detected or depth_variation_detected
    reasons = []
    if blink_detected:
        reasons.append("eye blink detected")
    if head_movement_detected:
        reasons.append("head movement detected")
    if depth_variation_detected:
        reasons.append("depth variation detected")

    if not reasons:
        reasons.append("awaiting liveness cues")

    return {
        "is_live": is_live,
        "message": ", ".join(reasons),
        "blink_detected": blink_detected,
        "head_movement_detected": head_movement_detected,
        "depth_variation_detected": depth_variation_detected,
    }
