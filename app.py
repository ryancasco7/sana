import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="BEED Department - Training Needs Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Domain names mapping
def get_domain_names():
    """Returns mapping of domain numbers to domain names"""
    return {
        '1': 'Teaching Strategies and Pedagogies',
        '2': 'Classroom Management Techniques',
        '3': 'Teaching Literacy and Numeracy in Early Grades',
        '4': 'Differentiated Instruction and Inclusive Education',
        '5': 'Integrating ICT in the Classroom',
        '6': 'Assessment and Evaluation of Learning',
        '7': 'Child Protection and Safe Learning Environment',
        '8': 'Parent and Community Engagement in Learning',
        '9': '21st Century Skills',
        '10': 'Values Education and Character Development',
        '11': 'Remediation Teaching Strategies',
        '12': 'Mental Health and Well-Being for Educators',
        '13': 'Curriculum Development and Planning'
    }

# Cluster interpretations
def get_cluster_interpretation(cluster_id):
    """Returns the interpretation/description for each cluster
    
    Note: Rating scale is 1-No Need, 2-Low Need, 3-Moderate Need, 4-High Need, 5-Urgent Need
    Higher scores indicate higher training needs, not higher proficiency.
    """
    interpretations = {
        0: "**Experienced Teachers with Lower Training Needs**: This cluster has the highest average age (38.56) and years of experience (13.80). They report generally lower training needs (lower ratings ~1.6-2.4) across teaching competencies including Inquiry-Based Learning, Project-Based Learning, Classroom Management, and Time/Stress Management. This suggests they perceive their skills as more proficient and have lower training needs compared to other groups. Despite high experience, they may still benefit from confidence-building and skill validation programs.",
        1: "**Moderately Proficient Experienced Teachers**: This cluster has high average age (37.51) and years of experience (12.55), similar to Clusters 0 and 3. They report moderate training needs across most competencies - higher than Cluster 0 but lower than Clusters 2 and 3 in many areas. This large group of experienced teachers feels reasonably competent and has moderate training needs across a broad range of teaching and professional development skills.",
        2: "**Younger Teachers with High Training Needs**: This cluster has the lowest average age (26.89) and years of experience (3.82). They report the highest training needs (highest ratings ~4.2-4.5) across most competencies, particularly in modern approaches like Inquiry-Based Learning (4.22), Project-Based Learning (4.23), Cooperative Learning (4.40), Classroom Management (4.26), Assessment (4.38), and Professional Development areas (Wellness programs 4.41, Peer Support 4.47). These younger, recently trained teachers recognize significant training needs and would benefit from comprehensive professional development programs.",
        3: "**Experienced Teachers with High Training Needs**: This cluster has high average age (37.18) and years of experience (12.53), similar to Clusters 0 and 1. They report consistently high training needs across most competencies (ratings ~3.5-3.7) including Inquiry-Based Learning (3.49), Project-Based Learning (3.56), Classroom Management (3.58), Assessment (3.66), and Professional Development (Wellness 3.67, Peer Support 3.67). These experienced teachers recognize substantial training needs and maintain awareness of skill development opportunities across pedagogical approaches."
    }
    return interpretations.get(cluster_id, "Cluster interpretation not available.")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('clustering_results.xlsx')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main app
def main():
    st.title("📊 Training Needs Analysis")
    st.markdown("### BEED Department Extension Program")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard", "Cluster Profiles", "Training Recommendations", "Self Assessment", "Admin Tools"]
    )
    
    if page == "Dashboard":
        show_dashboard(df)
    elif page == "Cluster Profiles":
        show_cluster_profiles(df)
    elif page == "Training Recommendations":
        show_recommendations(df)
    elif page == "Self Assessment":
        show_self_assessment(df)
    elif page == "Admin Tools":
        show_admin_tools(df)

def show_dashboard(df):
    st.header("📈 Overview Dashboard")
    
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Participants", len(df))
    
    with col2:
        st.metric("Number of Clusters", df['Cluster'].nunique())
    
    with col3:
        st.metric("Average Age", f"{df['Age'].mean():.1f}")
    
    with col4:
        st.metric("Years of Experience", f"{df['Years of Experience in Teaching'].mean():.1f}")
    
    st.divider()
    
    # Cluster interpretations overview
    st.subheader("📋 Cluster Interpretations")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_size = cluster_counts[cluster_id]
        interpretation = get_cluster_interpretation(cluster_id)
        with st.expander(f"Cluster {cluster_id} (n={cluster_size}) - Click to view interpretation", expanded=False):
            st.write(f"**Profile:** {interpretation}")
    
    st.divider()
    
    # Cluster distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cluster Distribution")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {i}" for i in cluster_counts.index],
            title="Participants per Cluster"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Cluster Size Comparison")
        fig_bar = px.bar(
            x=[f"Cluster {i}" for i in cluster_counts.index],
            y=cluster_counts.values,
            title="Number of Participants per Cluster",
            labels={"x": "Cluster", "y": "Count"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # PCA Visualization
    st.subheader("Cluster Visualization (PCA)")
    if st.checkbox("Show PCA Visualization"):
        # Prepare data for PCA
        feature_cols = [col for col in df.columns if col not in ['Cluster', 'Age', 'Gender', 'Years of Experience in Teaching']]
        
        if len(feature_cols) > 0:
            X = df[feature_cols].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_scaled)
            
            pca_df = pd.DataFrame(
                principal_components,
                columns=['PC1', 'PC2']
            )
            pca_df['Cluster'] = df['Cluster'].values
            
            fig_scatter = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                color_discrete_sequence=px.colors.qualitative.Set1,
                title="2D PCA Visualization of Clusters",
                labels={"PC1": f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})",
                       "PC2": f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})"}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.info(f"PCA explains {sum(pca.explained_variance_ratio_):.2%} of the variance")

def show_cluster_profiles(df):
    st.header("🔍 Cluster Profiles & Analysis")
    
    # Cluster selector
    selected_cluster = st.selectbox(
        "Select Cluster to Analyze",
        sorted(df['Cluster'].unique())
    )
    
    cluster_data = df[df['Cluster'] == selected_cluster]
    
    st.subheader(f"Cluster {selected_cluster} Profile")
    
    # Cluster interpretation
    interpretation = get_cluster_interpretation(selected_cluster)
    st.info(f"**Cluster {selected_cluster} Interpretation:** {interpretation}")
    
    # Show actual statistics
    st.markdown("**Key Characteristics:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Average Age:** {cluster_data['Age'].mean():.2f}")
    with col2:
        st.write(f"**Years of Experience:** {cluster_data['Years of Experience in Teaching'].mean():.2f}")
    with col3:
        avg_rating = cluster_data[[col for col in cluster_data.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.'))]].mean().mean()
        st.write(f"**Avg Training Need Rating:** {avg_rating:.2f}")
    
    st.divider()
    
    # Demographics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Participants", len(cluster_data))
        st.metric("Average Age", f"{cluster_data['Age'].mean():.1f}")
    
    with col2:
        gender_dist = cluster_data['Gender'].value_counts()
        st.metric("Male", gender_dist.get(1, 0))
        st.metric("Female", gender_dist.get(2, 0))
    
    with col3:
        st.metric("Avg. Years of Experience", f"{cluster_data['Years of Experience in Teaching'].mean():.1f}")
    
    st.divider()
    
    # Training needs analysis by domain
    st.subheader("Training Needs Analysis by Domain")
    
    # Show key competency highlights based on cluster interpretation
    st.markdown("**Key Competency Highlights:**")
    competency_cols = [col for col in cluster_data.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.'))]
    if len(competency_cols) > 0:
        top_needs = cluster_data[competency_cols].mean().sort_values(ascending=False).head(5)
        st.write("**Highest Training Needs (Top 5):**")
        for comp, rating in top_needs.items():
            comp_name = comp.split('. ', 1)[1] if '. ' in comp else comp
            if rating >= 4.0:
                st.error(f"- {comp_name}: {rating:.2f} (High/Urgent Need)")
            elif rating >= 3.0:
                st.warning(f"- {comp_name}: {rating:.2f} (Moderate Need)")
            else:
                st.info(f"- {comp_name}: {rating:.2f} (Low Need)")
    
    st.divider()
    
    # Group competencies by domain
    domain_mapping = {}
    for col in df.columns:
        if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.')):
            domain_num = col.split('.')[0]
            if domain_num not in domain_mapping:
                domain_mapping[domain_num] = []
            domain_mapping[domain_num].append(col)
    
    # Calculate domain averages
    domain_avgs = {}
    for domain, cols in domain_mapping.items():
        cluster_avg = cluster_data[cols].mean().mean()
        overall_avg = df[cols].mean().mean()
        domain_avgs[domain] = {
            'cluster_avg': cluster_avg,
            'overall_avg': overall_avg,
            'gap': cluster_avg - overall_avg
        }
    
    # Visualize domain comparisons
    domain_names = get_domain_names()
    domains = [domain_names.get(d, f"Domain {d}") for d in sorted(domain_mapping.keys(), key=int)]
    cluster_scores = [domain_avgs[d]['cluster_avg'] for d in sorted(domain_mapping.keys(), key=int)]
    overall_scores = [domain_avgs[d]['overall_avg'] for d in sorted(domain_mapping.keys(), key=int)]
    
    fig_domain = go.Figure()
    fig_domain.add_trace(go.Bar(
        name='Cluster Average',
        x=domains,
        y=cluster_scores,
        marker_color='lightblue'
    ))
    fig_domain.add_trace(go.Bar(
        name='Overall Average',
        x=domains,
        y=overall_scores,
        marker_color='lightcoral'
    ))
    fig_domain.update_layout(
        title="Training Needs by Domain Comparison",
        xaxis_title="Competency Domain",
        yaxis_title="Average Training Need Rating (1=No Need, 5=Urgent Need)",
        barmode='group',
        yaxis=dict(range=[1, 5])
    )
    st.plotly_chart(fig_domain, use_container_width=True)
    
    # Add rating scale info
    st.caption("Rating Scale: 1=No Need, 2=Low Need, 3=Moderate Need, 4=High Need, 5=Urgent Need")
    
    # Identify training needs (higher scores = higher needs)
    st.subheader("Identified Training Needs")
    domain_names = get_domain_names()
    gaps = [(d, domain_avgs[d]['gap']) for d in sorted(domain_mapping.keys(), key=int)]
    gaps_sorted_high = sorted(gaps, key=lambda x: x[1], reverse=True)  # Highest needs first
    
    st.write("**High Priority Training Needs (above overall average):**")
    high_needs_count = 0
    for domain, gap in gaps_sorted_high:
        if gap > 0:
            domain_num = int(domain)
            avg_rating = domain_avgs[domain]['cluster_avg']
            domain_name = domain_names.get(domain, f"Domain {domain}")
            if avg_rating >= 4:
                priority = "URGENT"
                st.error(f"{domain_name}: {gap:.2f} above average (Rating: {avg_rating:.2f} - {priority})")
            elif avg_rating >= 3:
                priority = "HIGH"
                st.warning(f"{domain_name}: {gap:.2f} above average (Rating: {avg_rating:.2f} - {priority})")
            else:
                st.info(f"{domain_name}: {gap:.2f} above average (Rating: {avg_rating:.2f})")
            high_needs_count += 1
            if high_needs_count >= 5:
                break
    
    if high_needs_count == 0:
        st.success("This cluster reports lower training needs across all domains compared to the overall average.")
    
    st.write("**Lower Priority Areas (below overall average):**")
    low_needs_count = 0
    for domain, gap in gaps_sorted_high[::-1]:  # Reverse to get lowest needs
        if gap < 0:
            domain_name = domain_names.get(domain, f"Domain {domain}")
            st.success(f"{domain_name}: {gap:.2f} below average (Lower training need)")
            low_needs_count += 1
            if low_needs_count >= 3:
                break

def show_recommendations(df):
    st.header("🎯 Training Program Recommendations")
    
    # Domain names mapping
    domain_names = get_domain_names()
    
    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(df['Cluster'].unique())
    )
    
    cluster_data = df[df['Cluster'] == selected_cluster]
    
    # Show cluster interpretation
    interpretation = get_cluster_interpretation(selected_cluster)
    st.info(f"**Cluster {selected_cluster} Profile:** {interpretation}")
    st.divider()
    
    # Group competencies by domain
    domain_mapping = {}
    for col in df.columns:
        if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.')):
            domain_num = col.split('.')[0]
            if domain_num not in domain_mapping:
                domain_mapping[domain_num] = []
            domain_mapping[domain_num].append(col)
    
    # Calculate training needs for each domain
    # Higher scores = higher training needs
    recommendations = []
    for domain, cols in sorted(domain_mapping.items(), key=lambda x: int(x[0])):
        cluster_avg = cluster_data[cols].mean().mean()
        overall_avg = df[cols].mean().mean()
        gap = cluster_avg - overall_avg
        
        # Recommend training when cluster has higher needs than average (gap > 0)
        if gap > 0.3 or cluster_avg >= 3.5:  # Significant need threshold
            if cluster_avg >= 4.5:
                priority = 'URGENT'
            elif cluster_avg >= 4.0:
                priority = 'HIGH'
            elif cluster_avg >= 3.5 or gap > 0.5:
                priority = 'MEDIUM'
            else:
                priority = 'LOW'
            
            recommendations.append({
                'domain': domain,
                'name': domain_names.get(domain, f"Domain {domain}"),
                'cluster_avg': cluster_avg,
                'gap': gap,
                'priority': priority,
                'competencies': cols
            })
    
    if recommendations:
        st.subheader(f"Recommended Training Programs for Cluster {selected_cluster}")
        st.caption("Recommendations are based on training need ratings: 1=No Need, 2=Low Need, 3=Moderate Need, 4=High Need, 5=Urgent Need")
        
        # Sort by priority and cluster average (highest needs first)
        recommendations_sorted = sorted(recommendations, key=lambda x: (x['cluster_avg'], x['gap']), reverse=True)
        
        for i, rec in enumerate(recommendations_sorted, 1):
            with st.expander(f"{i}. {rec['name']} - Priority: {rec['priority']} (Rating: {rec['cluster_avg']:.2f})", expanded=True):
                st.write(f"**Average Training Need Rating:** {rec['cluster_avg']:.2f}")
                st.write(f"**Gap vs Overall Average:** {rec['gap']:.2f} (Higher = More Need)")
                st.write(f"**Recommended Focus Areas:**")
                
                # Get specific competency needs within domain
                comp_needs = []
                for col in rec['competencies']:
                    cluster_avg = cluster_data[col].mean()
                    overall_avg = df[col].mean()
                    comp_needs.append((col, cluster_avg, cluster_avg - overall_avg))
                
                # Sort by cluster average (highest needs first)
                comp_needs_sorted = sorted(comp_needs, key=lambda x: x[1], reverse=True)
                for comp, comp_avg, gap_val in comp_needs_sorted[:5]:  # Top 5 needs
                    if comp_avg >= 3.0:  # Show moderate or higher needs
                        comp_name = comp.split('. ', 1)[1] if '. ' in comp else comp
                        if comp_avg >= 4.5:
                            need_level = "URGENT"
                            st.error(f"- {comp_name} (Rating: {comp_avg:.2f} - {need_level})")
                        elif comp_avg >= 4.0:
                            need_level = "HIGH"
                            st.warning(f"- {comp_name} (Rating: {comp_avg:.2f} - {need_level})")
                        else:
                            need_level = "MODERATE"
                            st.info(f"- {comp_name} (Rating: {comp_avg:.2f} - {need_level})")
                
                st.write("**Suggested Training Programs:**")
                if rec['domain'] == '1':
                    st.write("- Inquiry-Based Learning Workshop")
                    st.write("- Project-Based Learning Implementation")
                    st.write("- Contextualized Teaching Strategies")
                elif rec['domain'] == '2':
                    st.write("- Classroom Management Techniques")
                    st.write("- Positive Behavior Support Systems")
                    st.write("- Time Management Strategies")
                elif rec['domain'] == '5':
                    st.write("- Digital Literacy Training")
                    st.write("- LMS Implementation Workshop")
                    st.write("- Blended Learning Strategies")
                else:
                    st.write(f"- Training program for {rec['name']}")
                    st.write("- Skill development workshops")
                    st.write("- Hands-on practice sessions")
    else:
        st.info("No significant training needs identified for this cluster. This group reports lower training needs compared to the overall average.")

@st.cache_resource
def get_clustering_model(df):
    """Create and cache the KMeans clustering model"""
    # Get competency columns (all columns that start with domain numbers)
    competency_cols = [col for col in df.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.'))]
    
    # Prepare features for clustering (exclude demographics and cluster column)
    features = ['Age', 'Gender', 'Years of Experience in Teaching'] + competency_cols
    
    # Get feature data
    X = df[features].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit KMeans with k=4 (from the notebook)
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    return kmeans, scaler, features

def predict_cluster(new_data, kmeans, scaler, features):
    """Predict cluster for new assessment data"""
    # Create DataFrame with all features, filling missing ones with defaults
    feature_dict = {}
    for feature in features:
        if feature in new_data:
            feature_dict[feature] = new_data[feature]
        else:
            # Default values if missing
            if feature == 'Age':
                feature_dict[feature] = 30
            elif feature == 'Gender':
                feature_dict[feature] = 1
            elif feature == 'Years of Experience in Teaching':
                feature_dict[feature] = 5
            else:
                feature_dict[feature] = 3  # Default rating for competencies
    
    # Ensure features are in correct order
    X_new = pd.DataFrame([feature_dict])[features]
    
    # Handle missing values (shouldn't happen, but just in case)
    X_new = X_new.fillna(X_new.mean())
    
    # Scale the data
    X_new_scaled = scaler.transform(X_new)
    
    # Predict cluster
    cluster = kmeans.predict(X_new_scaled)[0]
    
    return cluster

def save_new_assessment(assessment_data):
    """Save new assessment to a separate database file"""
    # Create filename with timestamp
    filename = 'new_assessments.xlsx'
    
    # Try to load existing data
    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        # Convert assessment_data to DataFrame and append
        new_df = pd.DataFrame([assessment_data])
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # Create new DataFrame
        combined_df = pd.DataFrame([assessment_data])
    
    # Save to Excel
    combined_df.to_excel(filename, index=False)
    return filename


def show_self_assessment(df):
    st.header("📝 Self Assessment")
    st.markdown("Complete this assessment to find your cluster assignment and receive personalized training recommendations.")
    
    # Get competency columns
    competency_cols = [col for col in df.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.'))]
    
    # Get domain structure
    domain_mapping = {}
    for col in competency_cols:
        domain_num = col.split('.')[0]
        if domain_num not in domain_mapping:
            domain_mapping[domain_num] = []
        domain_mapping[domain_num].append(col)
    
    domain_names = get_domain_names()
    
    # Initialize session state for assessment data if not exists
    if 'assessment_submitted' not in st.session_state:
        st.session_state.assessment_submitted = False
        st.session_state.assessment_data = {}
        st.session_state.assessment_saved = False
    
    # Create form
    with st.form("self_assessment_form"):
        st.subheader("Personal Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=21, max_value=60, step=1)
        
        with col2:
            gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        
        with col3:
            years_experience = st.number_input("Years of Experience in Teaching", min_value=0, max_value=50, step=1)
        
        st.divider()
        
        # Competency ratings by domain
        st.subheader("Training Needs Assessment")
        st.info("**Rating Scale:** 1 = No Need, 2 = Low Need, 3 = Moderate Need, 4 = High Need, 5 = Urgent Need")
        
        # Create expandable sections for each domain
        for domain_num in sorted(domain_mapping.keys(), key=int):
            domain_cols_list = sorted(domain_mapping[domain_num])
            domain_name = domain_names.get(domain_num, f"Domain {domain_num}")
            
            with st.expander(f"🔰 {domain_num}. {domain_name}", expanded=True):
                for comp_col in domain_cols_list:
                    # Get competency name (after domain number)
                    comp_name = comp_col.split('. ', 1)[1] if '. ' in comp_col else comp_col
                    
                    # Display competency name
                    st.markdown(f"**{comp_name}**")
                    
                    # Create horizontal radio buttons (1-5 scale)
                    rating = st.radio(
                        "",
                        options=[1, 2, 3, 4, 5],
                        index=None,
                        key=f"form_{comp_col}",
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing between items
        
        submitted = st.form_submit_button("Submit Assessment")
        
        if submitted:
            # Collect all ratings from form widget values
            ratings = {}
            for comp_col in competency_cols:
                form_key = f"form_{comp_col}"
                if form_key in st.session_state:
                    ratings[comp_col] = st.session_state[form_key]
                else:
                    ratings[comp_col] = 3  # Default value if not set
            
            # Prepare assessment data
            assessment_data = {
                'Age': age,
                'Gender': gender,
                'Years of Experience in Teaching': years_experience,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            assessment_data.update(ratings)
            
            # Store in session state
            st.session_state.assessment_data = assessment_data
            st.session_state.assessment_submitted = True
            st.session_state.assessment_saved = False  # Reset save flag for new assessment
            
            # Rerun to show results
            st.rerun()
    
    # Display results after submission
    if st.session_state.assessment_submitted and st.session_state.assessment_data:
        # Add reset button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Start New Assessment"):
                st.session_state.assessment_submitted = False
                st.session_state.assessment_data = {}
                st.session_state.assessment_saved = False
                # Clear form widget values
                for comp_col in competency_cols:
                    form_key = f"form_{comp_col}"
                    if form_key in st.session_state:
                        del st.session_state[form_key]
                st.rerun()
        
        assessment_data = st.session_state.assessment_data.copy()
        
        # Get clustering model
        kmeans, scaler, features = get_clustering_model(df)
        
        # Predict cluster
        try:
            cluster = predict_cluster(assessment_data, kmeans, scaler, features)
            assessment_data['Cluster'] = cluster
            
            # Display results
            st.success("✅ Assessment submitted successfully!")
            st.divider()
            
            # Cluster assignment
            st.subheader(f"📊 Your Cluster Assignment: **Cluster {cluster}**")
            interpretation = get_cluster_interpretation(cluster)
            st.info(f"**Cluster {cluster} Profile:** {interpretation}")
            
            st.divider()
            
            # Training recommendations
            st.subheader("🎯 Your Training Recommendations")
            
            # Group by domain and calculate needs
            recommendations = []
            
            for domain_num in sorted(domain_mapping.keys(), key=int):
                domain_cols_list = domain_mapping[domain_num]
                domain_ratings = [assessment_data[col] for col in domain_cols_list]
                domain_avg = np.mean(domain_ratings)
                overall_avg = df[domain_cols_list].mean().mean()
                gap = domain_avg - overall_avg
                
                if domain_avg >= 3.5 or gap > 0.3:
                    if domain_avg >= 4.5:
                        priority = 'URGENT'
                    elif domain_avg >= 4.0:
                        priority = 'HIGH'
                    elif domain_avg >= 3.5 or gap > 0.5:
                        priority = 'MEDIUM'
                    else:
                        priority = 'LOW'
                    
                    recommendations.append({
                        'domain': domain_num,
                        'name': domain_names.get(domain_num, f"Domain {domain_num}"),
                        'avg_rating': domain_avg,
                        'gap': gap,
                        'priority': priority
                    })
            
            if recommendations:
                # Sort by priority and rating
                priority_order = {'URGENT': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
                recommendations.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['avg_rating']), reverse=True)
                
                for i, rec in enumerate(recommendations, 1):
                    if rec['priority'] == 'URGENT':
                        st.error(f"**{i}. {rec['name']}** - Priority: {rec['priority']} (Avg Rating: {rec['avg_rating']:.2f})")
                    elif rec['priority'] == 'HIGH':
                        st.warning(f"**{i}. {rec['name']}** - Priority: {rec['priority']} (Avg Rating: {rec['avg_rating']:.2f})")
                    else:
                        st.info(f"**{i}. {rec['name']}** - Priority: {rec['priority']} (Avg Rating: {rec['avg_rating']:.2f})")
                    
                    st.write(f"   - Your average rating: {rec['avg_rating']:.2f}")
                    st.write(f"   - Gap from overall average: {rec['gap']:.2f}")
                    st.write("")
            else:
                st.info("No significant training needs identified based on your assessment. Your ratings suggest you have lower training needs compared to the overall average.")
            
            # Save to database (only once)
            if not st.session_state.get('assessment_saved', False):
                try:
                    filename = save_new_assessment(assessment_data)
                    st.success(f"📁 Your assessment has been saved to `{filename}` for future study purposes.")
                    st.session_state.assessment_saved = True
                except Exception as e:
                    st.error(f"Error saving assessment: {e}")
            
        except Exception as e:
            st.error(f"Error predicting cluster: {e}")
            st.exception(e)

def show_admin_tools(df):
    st.header("⚙️ Administrative Tools")
    
    tab1, tab2, tab3 = st.tabs(["Data Management", "Export Visualizations", "Data Statistics"])
    
    with tab1:
        st.subheader("Dataset Information")
        st.write(f"**Total Records:** {len(df)}")
        st.write(f"**Total Features:** {len(df.columns)}")
        st.write(f"**Clusters:** {df['Cluster'].nunique()}")
        
        st.divider()
        
        st.subheader("Preview Data")
        num_rows = st.number_input("Number of rows to display", min_value=1, max_value=100, value=10)
        st.dataframe(df.head(num_rows), use_container_width=True)
        
        st.subheader("Download Data")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="clustering_results.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Export Visualizations")
        
        # Export options
        export_format = st.selectbox("Export Format", ["PNG", "PDF", "HTML"])
        
        if st.button("Generate All Visualizations"):
            st.info("Visualization export feature - implementation in progress")
    
    with tab3:
        st.subheader("Detailed Statistics")
        
        st.write("**Cluster Distribution:**")
        cluster_stats = df.groupby('Cluster').agg({
            'Age': ['mean', 'std', 'min', 'max'],
            'Years of Experience in Teaching': ['mean', 'std', 'min', 'max'],
            'Gender': lambda x: x.value_counts().to_dict()
        })
        st.dataframe(cluster_stats, use_container_width=True)
        
        st.write("**Missing Values:**")
        missing = df.isnull().sum()
        st.dataframe(missing[missing > 0], use_container_width=True)

if __name__ == "__main__":
    main()

