# Web-Based Clustering Analysis of Training Needs for BEED Department Extension Program

A comprehensive web application for visualizing and analyzing pre-computed clustering results of training needs data for the BEED Department Extension Program.

## Features

### 📊 Dashboard
- Overview metrics (total participants, clusters, demographics)
- Interactive cluster distribution visualizations
- 2D PCA visualization of cluster groupings
- Real-time data analysis

### 🔍 Cluster Profiles
- Detailed analysis of each cluster
- Participant demographics per cluster
- Competency domain analysis
- Skill gap identification
- Comparison with overall averages

### 🎯 Training Recommendations
- Automated training program recommendations based on cluster analysis
- Priority-based recommendations (High/Medium)
- Specific competency focus areas
- Suggested training programs per domain

### ⚙️ Admin Tools
- Dataset management and preview
- Data export capabilities
- Statistical summaries
- Visualization export options

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure `clustering_results.xlsx` is in the project directory
2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Access the application in your browser (default: http://localhost:8501)

## Data Format

The application expects an Excel file (`clustering_results.xlsx`) with:
- Participant demographics (Age, Gender, Years of Experience)
- 60+ competency/skill assessment columns organized into 13 domains
- A `Cluster` column indicating cluster assignments (0-3)

## Project Structure

```
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── clustering_results.xlsx     # Input data file
└── ClusteringRESULTS.ipynb     # Original clustering analysis notebook
```

## Competency Domains

1. Teaching Methodologies
2. Classroom Management
3. Literacy & Numeracy
4. Inclusive Education
5. Technology Integration
6. Assessment Techniques
7. Child Protection
8. Community Engagement
9. 21st Century Skills
10. Values Education
11. Remediation Strategies
12. Teacher Wellness
13. Curriculum Development

## License

This project is developed for the BEED Department Extension Program.

