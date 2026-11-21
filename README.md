🛡️ FakeNews Trust Analyzer

An AI-powered full-stack web application that detects fake news and assigns a trust/credibility score to news articles using Machine Learning and Natural Language Processing (NLP).

Python
React
Flask
NLP
License

📌 Features
🔍 Fake News Detection — Analyze any news article or headline using trained ML models
📊 Trust Score — Get a credibility/trust score (0–100%) for any news content
🌐 Web Scraping — Fetch and analyze live news from URLs automatically
🧠 NLP Processing — Deep text analysis using Natural Language Processing
🖥️ Web Dashboard — Clean React-based UI to interact with the analyzer
🔌 REST API — Expose endpoints for integration with other services
🗂️ Project Structure
FakeNewsTrustAnalyzer/
├── backend/                  # Python backend (Flask/FastAPI)
│   ├── app.py                # Main application entry point
│   ├── model/                # ML/NLP models
│   ├── routes/               # API route definitions
│   ├── scraper/              # News scraping logic
│   └── requirements.txt      # Python dependencies
│
├── frontend/                 # React / Next.js frontend
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   ├── pages/            # App pages
│   │   └── App.js            # Root component
│   ├── package.json
│   └── .env.example          # Frontend environment variables
│
├── .gitignore
└── README.md
⚙️ Prerequisites

Make sure the following are installed on your system:

Tool	Version	Link
Python	3.9+	python.org

Node.js	18+	nodejs.org

npm / yarn	latest	Comes with Node.js
Git	latest	git-scm.com
🚀 Getting Started — Clone & Run Locally
1️⃣ Clone the Repository
bash
git clone https://github.com/GautamTiwari2410031400/FakeNewsTrustAnalyzer.git
cd FakeNewsTrustAnalyzer
2️⃣ Setup the Backend (Python + Flask)
bash
# Navigate to the backend folder
cd backend
# Create a virtual environment
python -m venv venv
# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
# Run the backend server
python app.py

Backend will start at: http://localhost:5000

3️⃣ Setup the Frontend (React / Next.js)

Open a new terminal tab and run:

bash
# Navigate to the frontend folder
cd frontend
# Install Node.js dependencies
npm install
# Start the development server
npm start

Frontend will start at: http://localhost:3000

4️⃣ Open the App

Go to your browser and visit:

http://localhost:3000

You should see the FakeNews Trust Analyzer dashboard running! 🎉

🌐 Hosting on Another Server
Option A — Deploy on a Linux VPS (Ubuntu/Debian)
bash
# Update packages
sudo apt update && sudo apt upgrade -y
# Install Python, Node.js, Git
sudo apt install python3 python3-pip python3-venv nodejs npm git -y
# Clone the project
git clone https://github.com/GautamTiwari2410031400/FakeNewsTrustAnalyzer.git
cd FakeNewsTrustAnalyzer
# Setup Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py &        # run in background
# Setup Frontend (new terminal)
cd ../frontend
npm install
npm run build          # build for production
Option B — Run with Docker (Recommended for Server Deployment)
bash
# Clone the repo
git clone https://github.com/GautamTiwari2410031400/FakeNewsTrustAnalyzer.git
cd FakeNewsTrustAnalyzer
# Build and run with Docker Compose
docker-compose up --build

App will be available at: http://your-server-ip:3000

🔌 API Endpoints
Method	Endpoint	Description
POST	/api/analyze	Analyze text for fake news
POST	/api/scrape	Scrape and analyze a news URL
GET	/api/health	Health check
Example Request
bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking: Scientists discover cure for all diseases overnight"}'
Example Response
json
{
  "trust_score": 12.5,
  "verdict": "Likely Fake",
  "confidence": 0.87,
  "keywords": ["cure", "all diseases", "overnight"],
  "explanation": "Highly sensational language detected."
}
🧪 Running Tests
bash
# Backend tests
cd backend
pytest
# Frontend tests
cd frontend
npm test
🛠️ Environment Variables

Create a .env file in the backend/ folder:

env
FLASK_ENV=development
FLASK_PORT=5000
MODEL_PATH=model/fake_news_model.pkl

Create a .env file in the frontend/ folder:

env
REACT_APP_API_URL=http://localhost:5000
🤝 Contributing

Contributions are welcome! Here's how:

Fork the repository
Create a new branch: git checkout -b feature/your-feature
Commit your changes: git commit -m "Add your feature"
Push to your branch: git push origin feature/your-feature
Open a Pull Request
👨‍💻 Author

Gautam Tiwari

GitHub: @GautamTiwari2410031400
📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

⭐ Show Your Support

If you found this project helpful, please give it a ⭐ star on GitHub — it really helps!
