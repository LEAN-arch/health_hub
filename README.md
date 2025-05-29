# health_hub
“Community Health Intelligence Hub Streamlit App”
Health Hub Streamlit App

A community health intelligence dashboard for Community Health Workers (CHWs), Clinic Managers, and District Officers.

Features





Role-Based Dashboards: Tailored views for CHWs (referrals, alerts), Clinics (tests, supplies), and District Officers (disease trends, maps).



Visuals: Interactive Plotly charts (line, bar, donut, heatmap, choropleth map).



AI Analytics: Predictive symptom risks, supply forecasts, and anomaly detection.



Interactivity: Drill-downs, CSV export, and role selector.

Setup





Clone the repository:

git clone https://github.com/yourusername/health_hub.git
cd health_hub



Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install dependencies:

pip install -r requirements.txt



Run the app locally:

streamlit run app/app.py

Deployment





Push to GitHub:

git add .
git commit -m "Initial commit"
git push origin main



Deploy on Streamlit Community Cloud:





Log in to Streamlit Community Cloud.



Click "New app", select your repository, branch (main), and set the main file to app/app.py.



Deploy!

Structure





app/: Streamlit app entry point, pages, and utilities.



data/: Mock CSV and GeoJSON (tracked with Git LFS).



tests/: Unit tests for utilities.



.streamlit/: Configuration files.



scripts/: Setup scripts.

Dependencies

See requirements.txt. Key packages:





streamlit==1.38.0



pandas==2.2.2



plotly==5.22.0

Notes



