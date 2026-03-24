FLOW_STAGES = [
    {
        "id": "registration",
        "title": "Citizen Registration",
        "description": "Collect verified voter identity data, contact details, and enrollment photo.",
        "route": "/signup",
        "category": "onboarding",
    },
    {
        "id": "dataset",
        "title": "Face Dataset Creation",
        "description": "Store validated facial image records for recognition training and later verification.",
        "route": "/signup",
        "category": "biometric",
    },
    {
        "id": "training",
        "title": "Model Training (LBPH / CNN)",
        "description": "Train the active recognition model and track which biometric backend is in use.",
        "route": "/admin/dashboard",
        "category": "ai",
    },
    {
        "id": "database",
        "title": "Database Storage (MongoDB)",
        "description": "Persist voters, biometric metadata, model records, and election audit history.",
        "route": "/admin/dashboard",
        "category": "platform",
    },
    {
        "id": "authentication",
        "title": "Voting Day Authentication",
        "description": "Authenticate the voter with credentials and a second verification factor.",
        "route": "/login",
        "category": "security",
    },
    {
        "id": "verification",
        "title": "Face Verification",
        "description": "Match the live face against the enrolled voter identity before access is granted.",
        "route": "/face_verification",
        "category": "security",
    },
    {
        "id": "eligibility",
        "title": "Eligibility Check",
        "description": "Confirm the voter is registered, active, and has not already completed a ballot.",
        "route": "/admin/dashboard",
        "category": "rules",
    },
    {
        "id": "vote_casting",
        "title": "Secure Vote Casting",
        "description": "Open the protected vote-casting stage only after successful identity checks.",
        "route": "/admin/dashboard",
        "category": "ballot",
    },
    {
        "id": "encryption",
        "title": "Vote Encryption + Storage",
        "description": "Store encrypted ballot data with tamper-evident metadata and traceable audit entries.",
        "route": "/admin/dashboard",
        "category": "ballot",
    },
    {
        "id": "monitoring",
        "title": "Admin Monitoring Dashboard",
        "description": "Monitor the full election pipeline, model readiness, voter activity, and audit events.",
        "route": "/admin/dashboard",
        "category": "operations",
    },
]


ARCHITECTURE_COLLECTIONS = {
    "citizens": "voters",
    "face_datasets": "face_datasets",
    "trained_models": "trained_models",
    "encrypted_votes": "encrypted_votes",
    "election_audit_logs": "logs",
    "candidates": "candidates",
    "device_logs": "logs",
}


def get_flow_stages():
    return [stage.copy() for stage in FLOW_STAGES]
