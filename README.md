# Smart Voting System

Smart Voting System is a Flask-based election demo with voter registration, OTP login, guided face verification, vote casting, and an admin operations console backed by MongoDB.

## Highlights

- Voter registration with face dataset generation
- Aadhaar and password login with OTP verification
- Admin dashboard for voter, candidate, result, and log management
- MongoDB-backed storage for voters, votes, audit logs, and analytics
- Render-ready deployment configuration for presentation use

## Tech Stack

- Python
- Flask
- MongoDB
- OpenCV LBPH face recognition
- Gunicorn for production serving

## Project Structure

- `main.py` - Flask application entry point
- `mongo_db.py` - MongoDB access layer
- `biometric_modules.py` - face capture and dataset helpers
- `templates/` - Jinja templates
- `static/` - CSS, JS, and images
- `render.yaml` - Render deployment blueprint
- `wsgi.py` - production entry point for Gunicorn

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` values into your environment.
4. Start MongoDB locally, or provide a hosted MongoDB connection string.
5. Run the app:

```bash
python main.py
```

6. Open `http://127.0.0.1:5000`

## Important Environment Variables

- `FLASK_SECRET_KEY` - required for production sessions
- `ADMIN_USERNAME` - admin login username
- `ADMIN_PASSWORD` - admin login password
- `MONGODB_URI` - MongoDB connection string
- `MONGODB_DB_NAME` - database name
- `VOTE_ENCRYPTION_KEY` - vote encryption secret
- `PRESENTATION_MODE` - enables presentation-friendly deployment behavior
- `ALLOW_SERVER_CAMERA` - enables server-side OpenCV webcam streaming

## Admin Login

- URL: `/admin/login`
- Default username: `admin`
- Default password: `admin123`

Change those defaults before deploying.

## Presentation Mode on Render

This project originally streams the webcam from the server process through OpenCV. That works on a local machine with a webcam, but not on Render. For deployment, `render.yaml` enables:

- `PRESENTATION_MODE=true`
- `ALLOW_SERVER_CAMERA=false`

In this mode, the live server camera feed is disabled, but the guided verification flow still works for presentation and navigation demos.

## Health Check

- Endpoint: `/health`

Render uses this to confirm the web service is running.

## GitHub Push Checklist

Before pushing:

- Review `.gitignore`
- Confirm `.venv/`, `dataset/`, `static/uploads/`, and `otp.txt` are not tracked
- Do not commit real Aadhaar data, face images, MongoDB URIs, or production secrets
- Replace default admin credentials for any public demo

Suggested commands:

```bash
git init
git add .
git commit -m "Prepare smart voting system for GitHub and Render deployment"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Render Deployment

### Option 1: Blueprint deploy

1. Push this project to GitHub.
2. In Render, choose New + and select Blueprint.
3. Connect your GitHub repository.
4. Render will detect `render.yaml`.
5. Fill in the unsynced environment variables:
   - `ADMIN_USERNAME`
   - `ADMIN_PASSWORD`
   - `MONGODB_URI`
   - `VOTE_ENCRYPTION_KEY`
6. Deploy the service.

### Option 2: Manual web service

Use these values if you create the service manually:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn wsgi:application`
- Health check path: `/health`

Set these environment variables:

- `PYTHON_VERSION=3.14.3`
- `PRESENTATION_MODE=true`
- `ALLOW_SERVER_CAMERA=false`
- `FLASK_SECRET_KEY=<random-secret>`
- `ADMIN_USERNAME=<your-admin-user>`
- `ADMIN_PASSWORD=<your-admin-password>`
- `MONGODB_URI=<your-mongodb-uri>`
- `MONGODB_DB_NAME=smart_voting_system`
- `VOTE_ENCRYPTION_KEY=<random-secret>`

## Demo Flow for Presentation

For the smoothest Render presentation:

1. Open the home page.
2. Show voter registration and admin console screens.
3. Use a pre-created voter account for login.
4. Enter the OTP from the generated value if you are demoing locally.
5. Use the guided verification button to move into the ballot flow.
6. Show the admin dashboard, analytics, results, and logs.

## Notes

- Local webcam-based face streaming is intended for local demos only.
- Render file storage is ephemeral, so uploaded files and OTP text files are not persistent across restarts.
- A hosted MongoDB instance such as MongoDB Atlas is recommended for deployment.
