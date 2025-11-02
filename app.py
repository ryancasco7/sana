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
import hashlib
import json
import warnings
warnings.filterwarnings('ignore')

# Authentication functions
def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    users = {}
    
    # Default admin account
    default_admin = {
        'administrator': {
            'password': hash_password('admin123'),
            'role': 'admin',
            'created_at': '2024-01-01 00:00:00'
        }
    }
    
    if os.path.exists('users.json'):
        try:
            with open('users.json', 'r') as f:
                users = json.load(f)
        except:
            users = default_admin
    else:
        users = default_admin
    
    # Ensure admin account always exists
    if 'administrator' not in users:
        users['administrator'] = default_admin['administrator']
        save_users(users)
    
    return users

def save_users(users):
    """Save users to JSON file"""
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=4)

def signup(username, password):
    """Register a new user (always creates teacher account)"""
    users = load_users()
    
    if username in users:
        return False, "Username already exists"
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    users[username] = {
        'password': hash_password(password),
        'role': 'teacher',  # All new signups are teachers
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    save_users(users)
    return True, "Teacher account created successfully!"

def login(username, password):
    """Authenticate a user"""
    users = load_users()
    
    if username not in users:
        return False, "Username not found", None
    
    if users[username]['password'] != hash_password(password):
        return False, "Incorrect password", None
    
    role = users[username].get('role', 'teacher')
    return True, "Login successful!", role

def show_auth_page():
    """Display login/signup page"""
    
    # Custom CSS for auth page
    st.markdown("""
        <style>
        .auth-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 2rem;
        }
        .auth-header {
            text-align: center;
            color: #1E88E5;
            margin-bottom: 2rem;
        }
        .auth-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #0D47A1;
            margin-bottom: 0.5rem;
        }
        .auth-subtitle {
            font-size: 1.2rem;
            color: #424242;
            font-weight: 400;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.markdown('''
        <div class="auth-header">
            <div class="auth-title">📚 Training Needs Analyzer</div>
            <div class="auth-subtitle">BEED Department</div>
        </div>
    ''', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    with tab1:
        st.markdown("#### Login to Your Account")
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Login", type="primary", use_container_width=True)
            
            if submit:
                if username and password:
                    success, message, role = login(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_role = role
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter both username and password")
    
    with tab2:
        st.markdown("#### Create New Teacher Account")
        st.info("📌 All new accounts will be registered as Teacher accounts.")
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("signup_form"):
            new_username = st.text_input("Username", key="signup_username", placeholder="Choose a username (min. 3 characters)")
            new_password = st.text_input("Password", type="password", key="signup_password", placeholder="Choose a password (min. 6 characters)")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password", placeholder="Re-enter your password")
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Create Account", type="primary", use_container_width=True)
            
            if submit:
                if new_username and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("❌ Passwords do not match!")
                    else:
                        success, message = signup(new_username, new_password)
                        if success:
                            st.success(message)
                            st.info("✅ You can now login with your credentials")
                        else:
                            st.error(message)
                else:
                    st.warning("⚠️ Please fill in all fields")
    
    st.markdown('</div>', unsafe_allow_html=True)

def sanitize_df_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
    """Try to coerce dataframe columns to Arrow-friendly dtypes."""
    df = df.copy()
    for col in df.columns:
        try:
            if df[col].dtype == object:
                s = df[col].astype(str).str.strip()
                s_clean = s.str.replace(',', '', regex=False).str.replace('$', '', regex=False).str.replace('%', '', regex=False)
                num = pd.to_numeric(s_clean, errors='coerce')
                if num.notna().sum() >= max(1, int(0.5 * len(num))):
                    df[col] = num
                    continue
                dt = pd.to_datetime(s, errors='coerce')
                if dt.notna().sum() >= max(1, int(0.5 * len(dt))):
                    df[col] = dt
                    continue
            if pd.api.types.is_float_dtype(df[col].dtype):
                non_null = df[col].dropna()
                if not non_null.empty:
                    try:
                        if (non_null % 1 == 0).all():
                            df[col] = df[col].astype('Int64')
                    except Exception:
                        pass
        except Exception:
            continue
    if 'Total' in df.columns and df['Total'].dtype == object:
        try:
            tot = pd.to_numeric(df['Total'].astype(str).str.replace(',', '', regex=False), errors='coerce')
            if tot.notna().any():
                df['Total'] = tot.astype('Int64')
        except Exception:
            pass
    return df

st.set_page_config(
    page_title="BEED Department - Training Needs Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

def get_cluster_interpretation(cluster_id):
    """Returns the interpretation/description for each cluster"""
    interpretations = {
        0: "**Experienced Teachers with Lower Training Needs**: This cluster has the highest average age (38.56) and years of experience (13.80). They report generally lower training needs (lower ratings ~1.6-2.4) across teaching competencies including Inquiry-Based Learning, Project-Based Learning, Classroom Management, and Time/Stress Management. This suggests they perceive their skills as more proficient and have lower training needs compared to other groups. Despite high experience, they may still benefit from confidence-building and skill validation programs.",
        1: "**Moderately Proficient Experienced Teachers**: This cluster has high average age (37.51) and years of experience (12.55), similar to Clusters 0 and 3. They report moderate training needs across most competencies - higher than Cluster 0 but lower than Clusters 2 and 3 in many areas. This large group of experienced teachers feels reasonably competent and has moderate training needs across a broad range of teaching and professional development skills.",
        2: "**Younger Teachers with High Training Needs**: This cluster has the lowest average age (26.89) and years of experience (3.82). They report the highest training needs (highest ratings ~4.2-4.5) across most competencies, particularly in modern approaches like Inquiry-Based Learning (4.22), Project-Based Learning (4.23), Cooperative Learning (4.40), Classroom Management (4.26), Assessment (4.38), and Professional Development areas (Wellness programs 4.41, Peer Support 4.47). These younger, recently trained teachers recognize significant training needs and would benefit from comprehensive professional development programs.",
        3: "**Experienced Teachers with High Training Needs**: This cluster has high average age (37.18) and years of experience (12.53), similar to Clusters 0 and 1. They report consistently high training needs across most competencies (ratings ~3.5-3.7) including Inquiry-Based Learning (3.49), Project-Based Learning (3.56), Classroom Management (3.58), Assessment (3.66), and Professional Development (Wellness 3.67, Peer Support 3.67). These experienced teachers recognize substantial training needs and maintain awareness of skill development opportunities across pedagogical approaches."
    }
    return interpretations.get(cluster_id, "Cluster interpretation not available.")

@st.cache_data
def load_data():
    try:
        df = pd.read_excel('clustering_results.xlsx')
        try:
            df = sanitize_df_for_arrow(df)
        except Exception:
            pass
        return df
    except FileNotFoundError:
        st.error("❌ Error: 'clustering_results.xlsx' file not found. Please ensure the file is in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

def main():
    # Initialize session state for authentication
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    
    # Show auth page if not logged in
    if not st.session_state.logged_in:
        show_auth_page()
        return
    
    # Get user role
    user_role = st.session_state.user_role
    
    # Main app content (only shown when logged in)
    
    # Custom CSS for professional light blue theme
    st.markdown("""
        <style>
        /* Main theme colors */
        :root {
            --primary-blue: #1E88E5;
            --light-blue: #E3F2FD;
            --dark-blue: #0D47A1;
            --accent-blue: #42A5F5;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #1E88E5 0%, #42A5F5 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .main-title {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-align: center;
        }
        
        .main-subtitle {
            color: #E3F2FD;
            font-size: 1.2rem;
            text-align: center;
            margin-top: 0.5rem;
        }
        
        /* User info badge */
        .user-badge {
            background: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .user-role {
            font-weight: 600;
            color: #1E88E5;
            font-size: 0.9rem;
        }
        
        .user-name {
            color: #616161;
            font-size: 0.85rem;
        }
        
        /* Navigation buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        /* Cards and containers */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #1E88E5;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Section headers */
        h1, h2, h3 {
            color: #0D47A1;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #E3F2FD;
            border-radius: 5px;
            border-left: 3px solid #1E88E5;
        }
        
        /* Info boxes */
        .stAlert {
            border-radius: 8px;
        }
        
        /* Divider */
        hr {
            border-color: #BBDEFB;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header section
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('''
            <div class="main-header">
                <h1 class="main-title">📊 Training Needs Analysis System</h1>
                <p class="main-subtitle">Bachelor of Elementary Education Department</p>
            </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        role_icon = "🔑" if user_role == 'admin' else "👨‍🏫"
        role_text = "Administrator" if user_role == 'admin' else "Teacher"
        st.markdown(f'''
            <div class="user-badge">
                <div class="user-role">{role_icon} {role_text}</div>
                <div class="user-name">{st.session_state.username}</div>
            </div>
        ''', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.user_role = None
            st.rerun()
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Dashboard'
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigation based on role
    if user_role == 'admin':
        # Admin sees all buttons
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("Dashboard", key="nav_dashboard", use_container_width=True, type="secondary" if st.session_state.current_page != 'Dashboard' else "primary"):
                st.session_state.current_page = 'Dashboard'
                st.rerun()
        
        with col2:
            if st.button("Cluster Profiles", key="nav_profiles", use_container_width=True, type="secondary" if st.session_state.current_page != 'Cluster Profiles' else "primary"):
                st.session_state.current_page = 'Cluster Profiles'
                st.rerun()
        
        with col3:
            if st.button("Recommendations", key="nav_recommendations", use_container_width=True, type="secondary" if st.session_state.current_page != 'Training Recommendations' else "primary"):
                st.session_state.current_page = 'Training Recommendations'
                st.rerun()
        
        with col4:
            if st.button("Self Assessment", key="nav_assessment", use_container_width=True, type="secondary" if st.session_state.current_page != 'Self Assessment' else "primary"):
                st.session_state.current_page = 'Self Assessment'
                st.rerun()
        
        with col5:
            if st.button("Admin Tools", key="nav_admin", use_container_width=True, type="secondary" if st.session_state.current_page != 'Admin Tools' else "primary"):
                st.session_state.current_page = 'Admin Tools'
                st.rerun()
    else:
        # Teacher sees only Dashboard, Cluster Profiles, and Self Assessment
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Dashboard", key="nav_dashboard", use_container_width=True, type="secondary" if st.session_state.current_page != 'Dashboard' else "primary"):
                st.session_state.current_page = 'Dashboard'
                st.rerun()
        
        with col2:
            if st.button("Cluster Profiles", key="nav_profiles", use_container_width=True, type="secondary" if st.session_state.current_page != 'Cluster Profiles' else "primary"):
                st.session_state.current_page = 'Cluster Profiles'
                st.rerun()
        
        with col3:
            if st.button("Self Assessment", key="nav_assessment", use_container_width=True, type="secondary" if st.session_state.current_page != 'Self Assessment' else "primary"):
                st.session_state.current_page = 'Self Assessment'
                st.rerun()
    
    page = st.session_state.current_page
    
    # Check if teacher is trying to access restricted pages
    if user_role == 'teacher' and page in ['Training Recommendations', 'Admin Tools']:
        st.session_state.current_page = 'Dashboard'
        page = 'Dashboard'
    
    st.markdown("---")
    
    if page == "Dashboard":
        show_dashboard(df)
    elif page == "Cluster Profiles":
        show_cluster_profiles(df)
    elif page == "Training Recommendations":
        if user_role == 'admin':
            show_recommendations(df)
        else:
            st.error("🔒 Access Denied: This page is only available to administrators.")
    elif page == "Self Assessment":
        show_self_assessment(df)
    elif page == "Admin Tools":
        if user_role == 'admin':
            show_admin_tools(df)
        else:
            st.error("🔒 Access Denied: This page is only available to administrators.")

def show_dashboard(df):
    st.header("📈 Overview Dashboard")
    
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
    
    st.subheader("📋 Cluster Interpretations")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_size = cluster_counts[cluster_id]
        interpretation = get_cluster_interpretation(cluster_id)
        with st.expander(f"Cluster {cluster_id} (n={cluster_size}) - Click to view interpretation", expanded=False):
            st.write(f"**Profile:** {interpretation}")
    
    st.divider()
    
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
    
    st.subheader("Cluster Visualization (PCA)")
    if st.checkbox("Show PCA Visualization"):
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
                color_discrete_sequence=['#0D47A1', '#1E88E5', '#42A5F5', '#90CAF9'],
                title="2D PCA Visualization of Clusters",
                labels={"PC1": f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})",
                       "PC2": f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})"}
            )
            fig_scatter.update_layout(
                font=dict(size=12),
                title_font=dict(size=16, color='#0D47A1')
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.info(f"PCA explains {sum(pca.explained_variance_ratio_):.2%} of the variance")

def show_cluster_profiles(df):
    st.header("🔍 Cluster Profiles & Analysis")
    
    if 'selected_cluster' not in st.session_state:
        st.session_state.selected_cluster = sorted(df['Cluster'].unique())[0]
    
    st.markdown("**Select Cluster:**")
    clusters = sorted(df['Cluster'].unique())
    cols = st.columns(len(clusters))
    
    for i, cluster in enumerate(clusters):
        with cols[i]:
            if st.button(
                f"Cluster {cluster}",
                key=f"cluster_btn_{cluster}",
                use_container_width=True,
                type="primary" if st.session_state.selected_cluster == cluster else "secondary"
            ):
                st.session_state.selected_cluster = cluster
                st.rerun()
    
    selected_cluster = st.session_state.selected_cluster
    cluster_data = df[df['Cluster'] == selected_cluster]
    
    st.markdown("---")
    
    st.subheader(f"Cluster {selected_cluster} Profile")
    
    interpretation = get_cluster_interpretation(selected_cluster)
    st.info(f"**Cluster {selected_cluster} Interpretation:** {interpretation}")
    
    st.markdown("**Key Characteristics:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Average Age:** {cluster_data['Age'].mean():.2f}")
    with col2:
        st.write(f"**Years of Experience:** {cluster_data['Years of Experience in Teaching'].mean():.2f}")
    with col3:
        competency_cols = [col for col in cluster_data.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.'))]
        if competency_cols:
            avg_rating = cluster_data[competency_cols].mean().mean()
            st.write(f"**Avg Training Need Rating:** {avg_rating:.2f}")
    
    st.divider()
    
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
    
    st.subheader("Training Needs Analysis by Domain")
    
    competency_cols = [col for col in cluster_data.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.'))]
    if len(competency_cols) > 0:
        st.markdown("**Key Competency Highlights:**")
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
    
    domain_mapping = {}
    for col in df.columns:
        if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.')):
            domain_num = col.split('.')[0]
            if domain_num not in domain_mapping:
                domain_mapping[domain_num] = []
            domain_mapping[domain_num].append(col)
    
    domain_avgs = {}
    for domain, cols in domain_mapping.items():
        cluster_avg = cluster_data[cols].mean().mean()
        overall_avg = df[cols].mean().mean()
        domain_avgs[domain] = {
            'cluster_avg': cluster_avg,
            'overall_avg': overall_avg,
            'gap': cluster_avg - overall_avg
        }
    
    domain_names = get_domain_names()
    domains = [domain_names.get(d, f"Domain {d}") for d in sorted(domain_mapping.keys(), key=int)]
    cluster_scores = [domain_avgs[d]['cluster_avg'] for d in sorted(domain_mapping.keys(), key=int)]
    overall_scores = [domain_avgs[d]['overall_avg'] for d in sorted(domain_mapping.keys(), key=int)]
    
    fig_domain = go.Figure()
    fig_domain.add_trace(go.Bar(
        name='Cluster Average',
        x=domains,
        y=cluster_scores,
        marker_color='#1E88E5'
    ))
    fig_domain.add_trace(go.Bar(
        name='Overall Average',
        x=domains,
        y=overall_scores,
        marker_color='#90CAF9'
    ))
    fig_domain.update_layout(
        title="Training Needs by Domain Comparison",
        xaxis_title="Competency Domain",
        yaxis_title="Average Training Need Rating (1=No Need, 5=Urgent Need)",
        barmode='group',
        yaxis=dict(range=[1, 5]),
        height=500,
        font=dict(size=12),
        title_font=dict(size=16, color='#0D47A1')
    )
    st.plotly_chart(fig_domain, use_container_width=True)
    
    st.caption("Rating Scale: 1=No Need, 2=Low Need, 3=Moderate Need, 4=High Need, 5=Urgent Need")
    
    st.subheader("Identified Training Needs")
    gaps = [(d, domain_avgs[d]['gap']) for d in sorted(domain_mapping.keys(), key=int)]
    gaps_sorted_high = sorted(gaps, key=lambda x: x[1], reverse=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 High Priority Areas")
        st.markdown("*(Above overall average)*")
        high_needs_count = 0
        for domain, gap in gaps_sorted_high:
            if gap > 0:
                avg_rating = domain_avgs[domain]['cluster_avg']
                domain_name = domain_names.get(domain, f"Domain {domain}")
                if avg_rating >= 4:
                    priority = "URGENT"
                    st.error(f"**{domain_name}**\n\n+{gap:.2f} above average | Rating: {avg_rating:.2f} - {priority}")
                elif avg_rating >= 3:
                    priority = "HIGH"
                    st.warning(f"**{domain_name}**\n\n+{gap:.2f} above average | Rating: {avg_rating:.2f} - {priority}")
                else:
                    st.info(f"**{domain_name}**\n\n+{gap:.2f} above average | Rating: {avg_rating:.2f}")
                high_needs_count += 1
                if high_needs_count >= 5:
                    break
        
        if high_needs_count == 0:
            st.success("✅ No areas above overall average. This cluster shows lower training needs.")
    
    with col2:
        st.markdown("#### 🟢 Lower Priority Areas")
        st.markdown("*(Below overall average)*")
        low_needs_count = 0
        for domain, gap in gaps_sorted_high[::-1]:
            if gap < 0:
                domain_name = domain_names.get(domain, f"Domain {domain}")
                avg_rating = domain_avgs[domain]['cluster_avg']
                st.success(f"**{domain_name}**\n\n{gap:.2f} below average | Rating: {avg_rating:.2f}")
                low_needs_count += 1
                if low_needs_count >= 3:
                    break

def show_recommendations(df):
    st.markdown("## 🎯 Training Program Recommendations")
    st.markdown("<br>", unsafe_allow_html=True)
    
    domain_names = get_domain_names()
    
    if 'selected_cluster_rec' not in st.session_state:
        st.session_state.selected_cluster_rec = sorted(df['Cluster'].unique())[0]
    
    st.markdown("**Select Cluster:**")
    clusters = sorted(df['Cluster'].unique())
    cols = st.columns(len(clusters))
    
    for i, cluster in enumerate(clusters):
        with cols[i]:
            if st.button(
                f"Cluster {cluster}",
                key=f"cluster_rec_btn_{cluster}",
                use_container_width=True,
                type="primary" if st.session_state.selected_cluster_rec == cluster else "secondary"
            ):
                st.session_state.selected_cluster_rec = cluster
                st.rerun()
    
    selected_cluster = st.session_state.selected_cluster_rec
    cluster_data = df[df['Cluster'] == selected_cluster]
    
    st.markdown("---")
    
    interpretation = get_cluster_interpretation(selected_cluster)
    st.info(f"**Cluster {selected_cluster} Profile:** {interpretation}")
    st.divider()
    
    domain_mapping = {}
    for col in df.columns:
        if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.')):
            domain_num = col.split('.')[0]
            if domain_num not in domain_mapping:
                domain_mapping[domain_num] = []
            domain_mapping[domain_num].append(col)
    
    recommendations = []
    for domain, cols in sorted(domain_mapping.items(), key=lambda x: int(x[0])):
        cluster_avg = cluster_data[cols].mean().mean()
        overall_avg = df[cols].mean().mean()
        gap = cluster_avg - overall_avg
        
        if gap > 0.3 or cluster_avg >= 3.5:
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
        st.markdown(f"### Recommended Training Programs for Cluster {selected_cluster}")
        st.caption("📋 Based on training need ratings: 1=No Need, 2=Low Need, 3=Moderate Need, 4=High Need, 5=Urgent Need")
        st.markdown("<br>", unsafe_allow_html=True)
        
        recommendations_sorted = sorted(recommendations, key=lambda x: (x['cluster_avg'], x['gap']), reverse=True)
        
        for i, rec in enumerate(recommendations_sorted, 1):
            with st.expander(f"{i}. {rec['name']} - Priority: {rec['priority']} (Rating: {rec['cluster_avg']:.2f})", expanded=(i <= 3)):
                st.write(f"**Average Training Need Rating:** {rec['cluster_avg']:.2f}")
                st.write(f"**Gap vs Overall Average:** {rec['gap']:.2f} (Higher = More Need)")
                st.write(f"**Recommended Focus Areas:**")
                
                comp_needs = []
                for col in rec['competencies']:
                    cluster_avg = cluster_data[col].mean()
                    overall_avg = df[col].mean()
                    comp_needs.append((col, cluster_avg, cluster_avg - overall_avg))
                
                comp_needs_sorted = sorted(comp_needs, key=lambda x: x[1], reverse=True)
                for comp, comp_avg, gap_val in comp_needs_sorted[:5]:
                    if comp_avg >= 3.0:
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
    competency_cols = [col for col in df.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.'))]
    
    features = ['Age', 'Gender', 'Years of Experience in Teaching'] + competency_cols
    
    X = df[features].copy()
    X = X.fillna(X.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    return kmeans, scaler, features

def predict_cluster(new_data, kmeans, scaler, features):
    """Predict cluster for new assessment data"""
    feature_dict = {}
    for feature in features:
        if feature in new_data:
            feature_dict[feature] = new_data[feature]
        else:
            if feature == 'Age':
                feature_dict[feature] = 30
            elif feature == 'Gender':
                feature_dict[feature] = 1
            elif feature == 'Years of Experience in Teaching':
                feature_dict[feature] = 5
            else:
                feature_dict[feature] = 3
    
    X_new = pd.DataFrame([feature_dict])[features]
    X_new = X_new.fillna(X_new.mean())
    
    X_new_scaled = scaler.transform(X_new)
    cluster = kmeans.predict(X_new_scaled)[0]
    
    return cluster

def save_new_assessment(assessment_data):
    """Save new assessment to a separate database file"""
    filename = 'new_assessments.xlsx'
    
    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        new_df = pd.DataFrame([assessment_data])
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = pd.DataFrame([assessment_data])
    
    combined_df.to_excel(filename, index=False)
    return filename

def show_self_assessment(df):
    st.markdown("## 📝 Self Assessment")
    st.markdown("Complete this assessment to find your cluster assignment and receive personalized training recommendations.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    competency_cols = [col for col in df.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.'))]
    
    domain_mapping = {}
    for col in competency_cols:
        domain_num = col.split('.')[0]
        if domain_num not in domain_mapping:
            domain_mapping[domain_num] = []
        domain_mapping[domain_num].append(col)
    
    domain_names = get_domain_names()
    
    if 'assessment_submitted' not in st.session_state:
        st.session_state.assessment_submitted = False
        st.session_state.assessment_data = {}
        st.session_state.assessment_saved = False
    
    with st.form("self_assessment_form"):
        st.subheader("Personal Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=21, max_value=70, value=30, step=1)
        
        with col2:
            gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        
        with col3:
            years_experience = st.number_input("Years of Experience in Teaching", min_value=0, max_value=50, value=5, step=1)
        
        st.divider()
        
        st.markdown("#### Training Needs Assessment")
        st.info("**📊 Rating Scale:** 1 = No Need | 2 = Low Need | 3 = Moderate Need | 4 = High Need | 5 = Urgent Need")
        st.markdown("<br>", unsafe_allow_html=True)
        
        ratings = {}
        
        for domain_num in sorted(domain_mapping.keys(), key=int):
            domain_cols_list = sorted(domain_mapping[domain_num])
            domain_name = domain_names.get(domain_num, f"Domain {domain_num}")
            
            with st.expander(f"📚 {domain_num}. {domain_name}", expanded=(domain_num == '1')):
                for comp_col in domain_cols_list:
                    comp_name = comp_col.split('. ', 1)[1] if '. ' in comp_col else comp_col
                    
                    st.markdown(f"**{comp_name}**")
                    
                    rating = st.radio(
                        f"Rate: {comp_name}",
                        options=[1, 2, 3, 4, 5],
                        index=2,
                        key=f"form_{comp_col}",
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                    ratings[comp_col] = rating
                    st.markdown("<br>", unsafe_allow_html=True)
        
        submitted = st.form_submit_button("Submit Assessment", type="primary")
        
        if submitted:
            assessment_data = {
                'Age': age,
                'Gender': gender,
                'Years of Experience in Teaching': years_experience,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            assessment_data.update(ratings)
            
            st.session_state.assessment_data = assessment_data
            st.session_state.assessment_submitted = True
            st.session_state.assessment_saved = False
            
            st.rerun()
    
    if st.session_state.assessment_submitted and st.session_state.assessment_data:
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Start New Assessment"):
                st.session_state.assessment_submitted = False
                st.session_state.assessment_data = {}
                st.session_state.assessment_saved = False
                st.rerun()
        
        assessment_data = st.session_state.assessment_data.copy()
        
        kmeans, scaler, features = get_clustering_model(df)
        
        try:
            cluster = predict_cluster(assessment_data, kmeans, scaler, features)
            assessment_data['Cluster'] = cluster
            
            st.success("✅ Assessment submitted successfully!")
            st.divider()
            
            st.subheader(f"📊 Your Cluster Assignment: **Cluster {cluster}**")
            interpretation = get_cluster_interpretation(cluster)
            st.info(f"**Cluster {cluster} Profile:** {interpretation}")
            
            st.divider()
            
            st.subheader("🎯 Your Training Recommendations")
            
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
            
            if not st.session_state.get('assessment_saved', False):
                try:
                    filename = save_new_assessment(assessment_data)
                    st.session_state.assessment_saved = True
                except Exception as e:
                    pass
            
        except Exception as e:
            st.error(f"Error predicting cluster: {e}")
            st.write("Please check your inputs and try again.")

def show_admin_tools(df):
    st.markdown("## ⚙️ Administrative Tools")
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Data Management", "Export Visualizations", "Data Statistics"])
    
    with tab1:
        st.subheader("Dataset Information")
        st.write(f"**Total Records:** {len(df)}")
        st.write(f"**Total Features:** {len(df.columns)}")
        st.write(f"**Clusters:** {df['Cluster'].nunique()}")
        
        st.divider()
        
        st.subheader("Preview Data")
        num_rows = st.number_input("Number of rows to display", min_value=1, max_value=100, value=10, key="admin_rows")
        st.dataframe(df.head(num_rows), use_container_width=True)
        
        st.subheader("Download Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="clustering_results.csv",
            mime="text/csv",
            key="download_main_csv"
        )
        
        if os.path.exists('new_assessments.xlsx'):
            st.divider()
            st.subheader("New Assessments")
            try:
                new_assess_df = pd.read_excel('new_assessments.xlsx')
                st.write(f"**Total New Assessments:** {len(new_assess_df)}")
                st.dataframe(new_assess_df, use_container_width=True)
                
                csv_new = new_assess_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download New Assessments as CSV",
                    data=csv_new,
                    file_name="new_assessments.csv",
                    mime="text/csv",
                    key="download_new_csv"
                )
            except Exception as e:
                st.error(f"Error loading new assessments: {e}")
    
    with tab2:
        st.subheader("Export Visualizations")
        
        export_format = st.selectbox("Export Format", ["PNG", "PDF", "HTML"], key="export_format_select")
        
        if st.button("Generate All Visualizations", key="generate_viz_btn"):
            st.info("📊 Visualization export feature - implementation in progress")
            st.write("This feature will allow you to export:")
            st.write("- Cluster distribution charts")
            st.write("- PCA visualizations")
            st.write("- Domain comparison graphs")
            st.write("- Training needs heatmaps")
    
    with tab3:
        st.subheader("Detailed Statistics")
        
        st.write("**Cluster Distribution:**")
        cluster_stats = df.groupby('Cluster').agg({
            'Age': ['mean', 'std', 'min', 'max'],
            'Years of Experience in Teaching': ['mean', 'std', 'min', 'max']
        }).round(2)
        st.dataframe(cluster_stats, use_container_width=True)
        
        st.divider()
        
        st.write("**Gender Distribution by Cluster:**")
        gender_cluster = pd.crosstab(df['Cluster'], df['Gender'], margins=False)
        gender_cluster.columns = ['Male', 'Female']
        gender_cluster['Total'] = gender_cluster.sum(axis=1)
        totals = gender_cluster.sum()
        totals.name = 'Total'
        gender_cluster = pd.concat([gender_cluster, totals.to_frame().T])
        st.dataframe(gender_cluster, use_container_width=True)
        
        st.divider()
        
        st.write("**Missing Values Check:**")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.dataframe(missing[missing > 0], use_container_width=True)
        else:
            st.success("✅ No missing values in the dataset!")
        
        st.divider()
        
        st.write("**Competency Ratings Summary:**")
        competency_cols = [col for col in df.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.'))]
        if competency_cols:
            comp_stats = df[competency_cols].describe().round(2)
            st.dataframe(comp_stats, use_container_width=True)

if __name__ == "__main__":
    main()
