import os
import base64
from datetime import UTC, datetime
from typing import Dict, List, Optional
from uuid import uuid4

from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError
from dotenv import load_dotenv

from system_architecture import ARCHITECTURE_COLLECTIONS
from election_config import DEFAULT_CANDIDATES

load_dotenv()


MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "smart_voting_system")


_client: Optional[MongoClient] = None
_database = None
_collections: Dict[str, Collection] = {}


def get_database():
    global _client, _database

    if _database is None:
        _client = MongoClient(MONGODB_URI)
        _database = _client[MONGODB_DB_NAME]

    return _database


def get_collection(alias: str) -> Collection:
    if alias not in _collections:
        collection_name = ARCHITECTURE_COLLECTIONS[alias]
        _collections[alias] = get_database()[collection_name]
    return _collections[alias]


def get_users_collection() -> Collection:
    return get_collection("citizens")


def get_face_dataset_collection() -> Collection:
    return get_collection("face_datasets")


def get_trained_models_collection() -> Collection:
    return get_collection("trained_models")


def get_votes_collection() -> Collection:
    return get_collection("encrypted_votes")


def get_audit_logs_collection() -> Collection:
    return get_collection("election_audit_logs")


def get_candidates_collection() -> Collection:
    return get_collection("candidates")


def get_device_logs_collection() -> Collection:
    return get_collection("device_logs")


def initialize_database() -> None:
    users = get_users_collection()
    face_datasets = get_face_dataset_collection()
    trained_models = get_trained_models_collection()
    votes = get_votes_collection()
    audit_logs = get_audit_logs_collection()
    candidates = get_candidates_collection()

    users.create_index([("aadhaar", ASCENDING)], unique=True)
    users.create_index([("voter_id", ASCENDING)], unique=True)
    users.create_index([("phone", ASCENDING)])
    users.create_index([("email", ASCENDING)])
    users.create_index([("constituency", ASCENDING)])

    face_datasets.create_index([("aadhaar", ASCENDING)], unique=True)
    face_datasets.create_index([("created_at", DESCENDING)])

    trained_models.create_index([("model_name", ASCENDING), ("created_at", DESCENDING)])
    trained_models.create_index([("is_active", ASCENDING)])

    votes.create_index([("vote_id", ASCENDING)], unique=True)
    votes.create_index([("candidate_id", ASCENDING)])
    votes.create_index([("constituency", ASCENDING)])
    votes.create_index([("created_at", DESCENDING)])

    audit_logs.create_index([("log_id", ASCENDING)], unique=True)
    audit_logs.create_index([("voter_id", ASCENDING), ("timestamp", DESCENDING)])
    audit_logs.create_index([("event", ASCENDING), ("timestamp", DESCENDING)])
    audit_logs.create_index([("ip_address", ASCENDING), ("timestamp", DESCENDING)])

    candidates.create_index([("candidate_id", ASCENDING)], unique=True)
    candidates.create_index([("constituency", ASCENDING)])

    for candidate in DEFAULT_CANDIDATES:
        candidates.update_one(
            {"candidate_id": candidate["candidate_id"]},
            {
                "$setOnInsert": {
                    **candidate,
                    "constituency": "Vizag",
                    "created_at": datetime.now(UTC),
                }
            },
            upsert=True,
        )


def _read_file_base64(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as file_handle:
        return base64.b64encode(file_handle.read()).decode("utf-8")


def _read_dataset_images_base64(dataset_path: Optional[str]) -> List[str]:
    if not dataset_path or not os.path.isdir(dataset_path):
        return []

    encoded_images = []
    for file_name in sorted(os.listdir(dataset_path)):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        file_path = os.path.join(dataset_path, file_name)
        encoded = _read_file_base64(file_path)
        if encoded:
            encoded_images.append(encoded)
    return encoded_images


def get_face_records() -> List[Dict[str, str]]:
    datasets = get_face_dataset_collection()
    return list(
        datasets.find(
            {},
            {
                "_id": 0,
                "voter_id": 1,
                "aadhaar": 1,
                "image_path": 1,
                "face_dataset_path": 1,
                "profile_image_data": 1,
                "sample_images": 1,
            },
        )
    )


def log_audit_event(event_type: str, message: str, aadhar: Optional[str] = None) -> None:
    audit_logs = get_audit_logs_collection()
    voter = get_voter_by_aadhar(aadhar) if aadhar else None
    audit_logs.insert_one(
        {
            "log_id": f"LOG-{uuid4().hex[:12].upper()}",
            "voter_id": voter.get("voter_id") if voter else None,
            "event": f"{event_type}: {message}",
            "timestamp": datetime.now(UTC),
        }
    )


def update_model_status(model_name: str, sample_count: int, is_active: bool) -> None:
    trained_models = get_trained_models_collection()
    trained_models.update_one(
        {"model_name": model_name},
        {
            "$set": {
                "model_name": model_name,
                "sample_count": sample_count,
                "is_active": is_active,
                "updated_at": datetime.now(UTC),
            },
            "$setOnInsert": {"created_at": datetime.now(UTC)},
        },
        upsert=True,
    )


def get_system_metrics() -> Dict[str, object]:
    users = get_users_collection()
    face_datasets = get_face_dataset_collection()
    trained_models = get_trained_models_collection()
    votes = get_votes_collection()
    audit_logs = get_audit_logs_collection()

    latest_model = trained_models.find_one(
        {"is_active": True},
        sort=[("updated_at", DESCENDING), ("created_at", DESCENDING)],
        projection={"_id": 0, "model_name": 1, "sample_count": 1, "updated_at": 1},
    )

    latest_audit = audit_logs.find_one(
        {},
        sort=[("timestamp", DESCENDING)],
        projection={"_id": 0, "log_id": 1, "voter_id": 1, "event": 1, "timestamp": 1},
    )

    return {
        "registered_citizens": users.count_documents({}),
        "face_datasets": face_datasets.count_documents({}),
        "trained_models": trained_models.count_documents({}),
        "votes_stored": votes.count_documents({}),
        "audit_events": audit_logs.count_documents({}),
        "active_model": latest_model,
        "latest_audit": latest_audit,
        "device_logs": audit_logs.count_documents({"ip_address": {"$exists": True}}),
        "votes_cast": votes.count_documents({}),
        "remaining_voters": max(users.count_documents({}) - votes.count_documents({}), 0),
    }


def create_user(
    voter_id: str,
    name: str,
    email: str,
    phone: str,
    age: int,
    constituency: str,
    aadhar: str,
    password: str,
    image_path: str,
    face_dataset_path: str,
    sample_count: int,
    profile_image_data: Optional[str] = None,
    sample_images: Optional[List[str]] = None,
) -> bool:
    users = get_users_collection()
    face_datasets = get_face_dataset_collection()

    try:
        now = datetime.now(UTC)
        users.insert_one(
            {
                "voter_id": voter_id,
                "name": name,
                "email": email,
                "phone": phone,
                "age": age,
                "constituency": constituency,
                "aadhaar": aadhar,
                "password": password,
                "image_path": image_path,
                "face_dataset_path": face_dataset_path,
                "face_encoding": {
                    "model": "LBPH",
                    "dataset_path": face_dataset_path,
                },
                "profile_image_data": profile_image_data,
                "sample_images": sample_images or [],
                "registered": True,
                "has_voted": False,
                "registration_status": "registered",
                "eligibility_status": "pending_verification",
                "created_at": now,
            }
        )
        face_datasets.insert_one(
            {
                "voter_id": voter_id,
                "aadhaar": aadhar,
                "image_path": image_path,
                "face_dataset_path": face_dataset_path,
                "sample_count": sample_count,
                "profile_image_data": profile_image_data,
                "sample_images": sample_images or [],
                "dataset_status": "captured",
                "source": "registration_upload",
                "created_at": now,
            }
        )
        log_audit_event(
            "citizen_registered",
            "Citizen registration and face dataset capture completed.",
            aadhar,
        )
        return True
    except DuplicateKeyError:
        return False


def backfill_face_dataset_binaries() -> Dict[str, int]:
    users = get_users_collection()
    face_datasets = get_face_dataset_collection()
    updated = 0
    skipped = 0

    for dataset in face_datasets.find({}, {"_id": 0, "aadhaar": 1, "image_path": 1, "face_dataset_path": 1, "profile_image_data": 1, "sample_images": 1}):
        profile_image_data = dataset.get("profile_image_data")
        sample_images = dataset.get("sample_images") or []
        if profile_image_data and sample_images:
            skipped += 1
            continue

        encoded_profile = profile_image_data or _read_file_base64(dataset.get("image_path"))
        encoded_samples = sample_images or _read_dataset_images_base64(dataset.get("face_dataset_path"))
        if not encoded_profile and not encoded_samples:
            skipped += 1
            continue

        update_fields = {
            "profile_image_data": encoded_profile,
            "sample_images": encoded_samples,
        }
        face_datasets.update_one({"aadhaar": dataset["aadhaar"]}, {"$set": update_fields})
        users.update_one({"aadhaar": dataset["aadhaar"]}, {"$set": update_fields})
        updated += 1

    return {"updated": updated, "skipped": skipped}


def find_user_by_credentials(aadhar: str, password: str) -> Optional[Dict]:
    users = get_users_collection()
    return users.find_one({"aadhaar": aadhar, "password": password}, {"_id": 0})


def get_voter_by_aadhar(aadhar: str) -> Optional[Dict]:
    users = get_users_collection()
    return users.find_one({"aadhaar": aadhar}, {"_id": 0})


def get_voter_by_voter_id(voter_id: str) -> Optional[Dict]:
    users = get_users_collection()
    return users.find_one({"voter_id": voter_id}, {"_id": 0})


def check_voting_status(voter_id: str) -> Dict[str, object]:
    voter = get_voter_by_voter_id(voter_id)
    if not voter:
        return {
            "allowed": False,
            "message": "Vote denied - voter record not found.",
            "voter": None,
        }

    if not voter.get("registered"):
        return {
            "allowed": False,
            "message": "Vote denied - voter is not registered.",
            "voter": voter,
        }

    if voter.get("has_voted"):
        return {
            "allowed": False,
            "message": "Vote denied - duplicate voting prevented.",
            "voter": voter,
        }

    return {
        "allowed": True,
        "message": "Vote allowed - voter is eligible and has not voted yet.",
        "voter": voter,
    }


def get_registration_overview(limit: int = 5) -> List[Dict]:
    users = get_users_collection()
    return list(
        users.find(
            {},
            {
                "_id": 0,
                "voter_id": 1,
                "name": 1,
                "constituency": 1,
                "registered": 1,
                "has_voted": 1,
                "face_dataset_path": 1,
            },
        ).sort("created_at", DESCENDING).limit(limit)
    )


def get_all_voters() -> List[Dict]:
    users = get_users_collection()
    return list(
        users.find(
            {},
            {
                "_id": 0,
                "voter_id": 1,
                "name": 1,
                "email": 1,
                "phone": 1,
                "constituency": 1,
                "registered": 1,
                "has_voted": 1,
                "eligibility_status": 1,
                "created_at": 1,
            },
        ).sort([("created_at", DESCENDING), ("name", ASCENDING)])
    )


def update_voter_record(
    voter_id: str,
    name: str,
    email: str,
    phone: str,
    constituency: str,
    registered: bool,
) -> bool:
    users = get_users_collection()
    result = users.update_one(
        {"voter_id": voter_id},
        {
            "$set": {
                "name": name,
                "email": email,
                "phone": phone,
                "constituency": constituency,
                "registered": registered,
                "updated_at": datetime.now(UTC),
            }
        },
    )
    return result.matched_count > 0


def delete_voter_record(voter_id: str) -> bool:
    users = get_users_collection()
    face_datasets = get_face_dataset_collection()
    user = users.find_one({"voter_id": voter_id}, {"_id": 0, "aadhaar": 1})
    result = users.delete_one({"voter_id": voter_id})
    if result.deleted_count and user:
        face_datasets.delete_many({"aadhaar": user.get("aadhaar")})
    return result.deleted_count > 0


def get_candidate_list() -> List[Dict]:
    candidates = get_candidates_collection()
    return list(
        candidates.find(
            {},
            {"_id": 0, "candidate_id": 1, "name": 1, "party": 1, "constituency": 1},
        ).sort([("constituency", ASCENDING), ("name", ASCENDING)])
    )


def get_available_constituencies() -> List[str]:
    users = get_users_collection()
    candidates = get_candidates_collection()
    constituencies = set(users.distinct("constituency")) | set(candidates.distinct("constituency"))
    return sorted(value for value in constituencies if value)


def add_candidate(candidate_id: str, name: str, party: str, constituency: str) -> None:
    candidates = get_candidates_collection()
    candidates.update_one(
        {"candidate_id": candidate_id},
        {
            "$set": {
                "candidate_id": candidate_id,
                "name": name,
                "party": party,
                "constituency": constituency,
                "updated_at": datetime.now(UTC),
            },
            "$setOnInsert": {"created_at": datetime.now(UTC)},
        },
        upsert=True,
    )


def update_candidate(candidate_id: str, name: str, party: str, constituency: str) -> bool:
    candidates = get_candidates_collection()
    result = candidates.update_one(
        {"candidate_id": candidate_id},
        {
            "$set": {
                "name": name,
                "party": party,
                "constituency": constituency,
                "updated_at": datetime.now(UTC),
            }
        },
    )
    return result.matched_count > 0


def remove_candidate(candidate_id: str) -> bool:
    candidates = get_candidates_collection()
    result = candidates.delete_one({"candidate_id": candidate_id})
    return result.deleted_count > 0


def store_vote(voter_id: str, vote_id: str, constituency: str, candidate_id: str) -> Dict[str, object]:
    users = get_users_collection()
    votes = get_votes_collection()

    voter = users.find_one_and_update(
        {"voter_id": voter_id, "registered": True, "has_voted": False},
        {"$set": {"has_voted": True, "updated_at": datetime.now(UTC)}},
        projection={"_id": 0, "voter_id": 1, "aadhaar": 1, "constituency": 1, "has_voted": 1},
        return_document=ReturnDocument.AFTER,
    )
    if not voter:
        return {
            "stored": False,
            "message": "Vote denied - voter is not eligible to cast a ballot.",
            "vote": None,
        }

    vote_record = {
        "vote_id": vote_id,
        "constituency": constituency,
        "candidate_id": candidate_id,
        "timestamp": datetime.now(UTC),
        "created_at": datetime.now(UTC),
    }

    try:
        votes.insert_one(vote_record)
    except Exception:
        users.update_one({"voter_id": voter_id}, {"$set": {"has_voted": False}})
        raise

    return {
        "stored": True,
        "message": "Vote stored securely and voter status updated.",
        "vote": vote_record,
    }


def get_recent_votes(limit: int = 10) -> List[Dict]:
    votes = get_votes_collection()
    return list(
        votes.find(
            {},
            {"_id": 0, "vote_id": 1, "constituency": 1, "candidate_id": 1, "timestamp": 1},
        ).sort("created_at", DESCENDING).limit(limit)
    )


def get_vote_results() -> List[Dict]:
    vote_records = get_recent_votes(limit=100000)
    candidates = {candidate["candidate_id"]: candidate for candidate in get_candidate_list()}
    tallies: Dict[str, Dict[str, object]] = {}

    for vote in vote_records:
        candidate_id = vote["candidate_id"]
        candidate = candidates.get(
            candidate_id,
            {"name": candidate_id, "party": "Independent", "constituency": vote["constituency"]},
        )
        if candidate_id not in tallies:
            tallies[candidate_id] = {
                "candidate_id": candidate_id,
                "name": candidate["name"],
                "party": candidate["party"],
                "constituency": vote["constituency"],
                "votes": 0,
            }
        tallies[candidate_id]["votes"] += 1

    return sorted(tallies.values(), key=lambda row: (-row["votes"], row["name"]))


def log_device_activity(voter_id: str, ip_address: str, machine_id: str, event_type: str) -> None:
    device_logs = get_device_logs_collection()
    device_logs.insert_one(
        {
            "log_id": f"LOG-{uuid4().hex[:12].upper()}",
            "voter_id": voter_id,
            "event": event_type,
            "ip_address": ip_address,
            "machine_id": machine_id,
            "timestamp": datetime.now(UTC),
        }
    )


def get_recent_device_logs(limit: int = 10) -> List[Dict]:
    device_logs = get_device_logs_collection()
    return list(
        device_logs.find(
            {},
            {"_id": 0, "log_id": 1, "voter_id": 1, "event": 1, "ip_address": 1, "machine_id": 1, "timestamp": 1},
        ).sort("timestamp", DESCENDING).limit(limit)
    )


def reset_election() -> Dict[str, int]:
    users = get_users_collection()
    votes = get_votes_collection()

    updated_users = users.update_many({}, {"$set": {"has_voted": False, "updated_at": datetime.now(UTC)}})
    deleted_votes = votes.delete_many({})
    return {
        "reset_voters": updated_users.modified_count,
        "deleted_votes": deleted_votes.deleted_count,
    }
