import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Personality Predictor - AI Model",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    .extrovert-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load the model and preprocessor
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_comprehensive_model.pkl')
        preprocessor = joblib.load('preprocessor_comprehensive.pkl')
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load model
model, preprocessor = load_model()

if model is None or preprocessor is None:
    st.error("❌ Failed to load the model. Please check if the model files are present.")
    st.stop()

# Header
st.markdown('<h1 class="main-header">🧠 Personality Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Personality Analysis with 96.9% Accuracy</p>', unsafe_allow_html=True)

# Statistics cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: #667eea; font-size: 2rem;">96.9%</h2>
        <p>Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: #667eea; font-size: 2rem;">7</h2>
        <p>Input Features</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: #667eea; font-size: 2rem;">2</h2>
        <p>Personality Types</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: #667eea; font-size: 2rem;">18K+</h2>
        <p>Training Samples</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Input form
st.markdown("## 📝 Enter Your Behavioral Data")

col1, col2 = st.columns(2)

with col1:
    time_alone = st.number_input(
        "Time Spent Alone (hours per day)",
        min_value=0.0,
        max_value=24.0,
        value=4.0,
        step=0.5,
        help="How many hours do you spend alone each day?"
    )
    
    social_events = st.number_input(
        "Social Event Attendance (per month)",
        min_value=0,
        max_value=31,
        value=5,
        help="How many social events do you attend per month?"
    )
    
    going_outside = st.number_input(
        "Going Outside (times per week)",
        min_value=0,
        max_value=7,
        value=3,
        help="How many times do you go outside per week?"
    )
    
    friends_circle = st.number_input(
        "Friends Circle Size",
        min_value=0,
        max_value=100,
        value=8,
        help="How many close friends do you have?"
    )

with col2:
    post_frequency = st.number_input(
        "Post Frequency (per week)",
        min_value=0,
        max_value=50,
        value=3,
        help="How often do you post on social media per week?"
    )
    
    stage_fear = st.selectbox(
        "Stage Fear",
        ["No", "Yes"],
        help="Do you experience stage fear or public speaking anxiety?"
    )
    
    drained_socializing = st.selectbox(
        "Drained After Socializing",
        ["No", "Yes"],
        help="Do you feel drained after socializing with others?"
    )

# Prediction button
if st.button("🚀 Get My Personality Prediction", type="primary", use_container_width=True):
    with st.spinner("Analyzing your personality patterns..."):
        try:
            # Create input data
            input_data = {
                'Time_spent_Alone': [time_alone],
                'Stage_fear': [stage_fear],
                'Social_event_attendance': [social_events],
                'Going_outside': [going_outside],
                'Drained_after_socializing': [drained_socializing],
                'Friends_circle_size': [friends_circle],
                'Post_frequency': [post_frequency]
            }
            
            # Create DataFrame
            input_df = pd.DataFrame(input_data)
            
            # Preprocess
            processed_data = preprocessor.transform(input_df)
            
            # Make prediction
            prediction_proba = model.predict_proba(processed_data)[0]
            prediction = model.predict(processed_data)[0]
            
            # Map prediction
            personality_type = "Extrovert" if prediction == 0 else "Introvert"
            confidence = prediction_proba[prediction]
            
            # Display result
            st.markdown("---")
            
            if personality_type == "Extrovert":
                st.markdown(f"""
                <div class="result-card extrovert-result">
                    <h1 style="font-size: 3rem; margin-bottom: 1rem;">🎉 Extrovert</h1>
                    <h2 style="font-size: 2rem; margin-bottom: 1rem;">Confidence: {confidence*100:.1f}%</h2>
                    <p style="font-size: 1.2rem;">Based on your behavioral patterns, you are likely an extrovert!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card">
                    <h1 style="font-size: 3rem; margin-bottom: 1rem;">🧘 Introvert</h1>
                    <h2 style="font-size: 2rem; margin-bottom: 1rem;">Confidence: {confidence*100:.1f}%</h2>
                    <p style="font-size: 1.2rem;">Based on your behavioral patterns, you are likely an introvert!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence bar
            st.progress(confidence)
            st.caption(f"Confidence Level: {confidence*100:.1f}%")
            
            # Additional insights
            st.markdown("### 💡 Personality Insights")
            
            if personality_type == "Extrovert":
                st.markdown("""
                **Extrovert Traits:**
                - You gain energy from social interactions
                - You enjoy being around people
                - You're comfortable in social situations
                - You prefer group activities
                - You're outgoing and expressive
                """)
            else:
                st.markdown("""
                **Introvert Traits:**
                - You gain energy from alone time
                - You prefer deep, meaningful conversations
                - You need time to recharge after socializing
                - You enjoy solitary activities
                - You're thoughtful and reflective
                """)
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

# Sidebar with information
with st.sidebar:
    st.markdown("## 📊 About This Model")
    st.markdown("""
    This personality predictor uses a **Gradient Boosting Classifier** trained on **18,526 samples** with **96.9% accuracy**.
    
    **Features analyzed:**
    - Time spent alone
    - Social event attendance
    - Going outside frequency
    - Friends circle size
    - Social media posting
    - Stage fear
    - Energy after socializing
    """)
    
    st.markdown("## 🎯 How to Use")
    st.markdown("""
    1. Fill in your behavioral patterns
    2. Click the prediction button
    3. Get your personality type with confidence score
    4. Read insights about your personality
    """)
    
    st.markdown("## 🔧 Technical Details")
    st.markdown("""
    - **Algorithm**: Gradient Boosting Classifier
    - **Accuracy**: 96.9%
    - **Training Data**: 18,526 samples
    - **Features**: 7 behavioral patterns
    - **Framework**: Streamlit + scikit-learn
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Powered by Machine Learning • Built with Streamlit • 96.9% Accuracy</p>
    <p>🧠 AI-Powered Personality Analysis</p>
</div>
""", unsafe_allow_html=True) 