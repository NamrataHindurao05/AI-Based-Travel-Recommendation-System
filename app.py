import streamlit as st
import pandas as pd
import pickle
import numpy as np
from google import genai
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go


# Load environment variables
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="AI Travel Recommender",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS - FIXED WITH VISIBLE TEXT COLORS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .destination-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* FIXED CHATBOT STYLING - DARK TEXT ON LIGHT BACKGROUND */
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        line-height: 1.6;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        color: #1565C0 !important;
    }
    .user-message strong {
        color: #0D47A1 !important;
        font-size: 1.1rem;
    }
    .bot-message {
        background-color: #F1F8E9;
        border-left: 5px solid #8BC34A;
        color: #33691E !important;
    }
    .bot-message strong {
        color: #1B5E20 !important;
        font-size: 1.1rem;
    }
    
    /* Ensure all text inside chat is visible */
    .chat-message p {
        color: inherit !important;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# Load ML Model
@st.cache_resource
def load_model():
    try:
        with open('models/recommendation_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ Model not found! Run: python train_model.py")
        st.stop()


# Load Dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv('dataset/travel_data.csv')
    except FileNotFoundError:
        st.error("⚠️ Dataset not found! Run: python dataset/create_dataset.py")
        st.stop()


# Initialize Gemini AI with YOUR API KEY

GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"

@st.cache_resource
def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.warning("⚠️ GEMINI_API_KEY not found. Check your .env file.")
        return None

    try:
        client = genai.Client(api_key=api_key)
        print(f"✅ Gemini client initialized for {GEMINI_MODEL_NAME}")
        return client

    except Exception as e:
        st.warning(f"⚠️ Gemini client error: {e}")
        return None


gemini_model = init_gemini()


def get_gemini_insights(preferences, destinations):
    """Get AI-powered travel insights"""
    if not gemini_model:
        return "🔑 **Gemini AI not configured properly.**"
    
    try:
        dest_list = ', '.join([d['city'] for d in destinations[:3]])
        prompt = f"""
        As a friendly travel expert, provide personalized insights for these destinations: {dest_list}
        
        User preferences:
        - Destination Type: {preferences['type']}
        - Budget: {preferences['budget']}
        - Interests: Culture ({preferences['culture']}/10), Adventure ({preferences['adventure']}/10), Nature ({preferences['nature']}/10)
        
        Please provide:
        1. Why these destinations match their preferences (2-3 sentences)
        2. Best time to visit each destination
        3. One insider travel tip for each destination
        
        Keep your response engaging, practical, and under 200 words.
        """
        
        response = gemini_model.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt
        )
        return response.text
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ Gemini's free-tier quota is exhausted. Please try again after the quota resets."
        return f"⚠️ AI insights error: {str(e)}"


def chat_with_gemini(user_message, df):
    """Chat with Gemini AI about travel"""
    if not gemini_model:
        return "🔑 Gemini AI not available."
    
    try:
        destinations = ', '.join(df['city'].head(30).tolist())
        types = ', '.join(df['type'].unique().tolist())
        
        prompt = f"""
        You are a friendly, knowledgeable travel expert chatbot.
        
        Available destinations: {destinations}
        Destination types: {types}
        
        User question: {user_message}
        
        Provide helpful, specific, and engaging travel advice. If asking about destinations, 
        recommend from our available destinations. Be conversational and practical.
        Keep response between 100-150 words.
        """
        
        response = gemini_model.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt
        )
        return response.text
    
    except Exception as e:
        return f"⚠️ Chat error: {str(e)}"


def get_ml_recommendations(user_prefs, model_artifacts, top_n=5):
    """Get ML-based recommendations"""
    df = model_artifacts['dataset']
    scaler = model_artifacts['scaler']
    knn_model = model_artifacts['knn_model']
    feature_columns = model_artifacts['feature_columns']
    label_encoders = model_artifacts['label_encoders']
    
    try:
        type_encoded = label_encoders['type'].transform([user_prefs['type']])[0]
        cost_encoded = label_encoders['cost'].transform([user_prefs['budget']])[0]
    except:
        type_encoded = 0
        cost_encoded = 1
    
    user_features = {
        'avg_temp': user_prefs['temperature'],
        'safety_rating': user_prefs['safety'] / 2,
        'tourist_rating': 4.0,
        'nightlife_score': user_prefs['nightlife'],
        'culture_score': user_prefs['culture'],
        'nature_score': user_prefs['nature'],
        'adventure_score': user_prefs['adventure'],
        'family_friendly': user_prefs['family_friendly'],
        'food_rating': 4.0,
        'type_encoded': type_encoded,
        'cost_encoded': cost_encoded,
        'season_encoded': 0
    }
    
    user_array = pd.DataFrame(
        [[user_features[col] for col in feature_columns]],
        columns=feature_columns
    )
    user_scaled = scaler.transform(user_array)
    
    distances, indices = knn_model.kneighbors(user_scaled, n_neighbors=min(top_n, len(df)))
    
    recommendations = []
    for idx, dist in zip(indices[0], distances[0]):
        rec = df.iloc[idx].to_dict()
        rec['similarity_score'] = (1 - dist) * 100
        recommendations.append(rec)
    
    return recommendations


def main():
    st.markdown('<div class="main-header">✈️ AI Travel Recommender</div>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #555;'>Discover Your Perfect Destination with ML & Gemini AI</h3>", unsafe_allow_html=True)
    
    model_artifacts = load_model()
    df = load_data()
    
    # Model info
    with st.expander("ℹ️ About This Project"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Destinations", len(df))
        with col2:
            st.metric("🎯 Algorithm", "K-NN + RF")
        with col3:
            perf = model_artifacts.get('performance', {})
            r2 = perf.get('r2_score', 0)
            st.metric("📈 Model R²", f"{r2:.3f}" if r2 > 0 else "N/A")
        with col4:
            ai_status = "🟢 Active" if gemini_model else "🔴 Inactive"
            st.metric("🤖 AI Status", ai_status)
    
    # Sidebar
    with st.sidebar:
        st.title("🎯 Travel Preferences")
        st.markdown("---")
        
        destination_type = st.selectbox("🏖️ Destination Type", sorted(df['type'].unique()))
        budget = st.selectbox("💰 Budget", ['Budget', 'Mid-range', 'Luxury'], index=1)
        temperature = st.slider("🌡️ Temperature (°C)", 15, 35, 25)
        
        st.markdown("#### 📊 Interest Levels")
        col1, col2 = st.columns(2)
        with col1:
            nightlife = st.slider("🌃 Nightlife", 1, 10, 5)
            culture = st.slider("🎭 Culture", 1, 10, 7)
            nature = st.slider("🌲 Nature", 1, 10, 6)
        with col2:
            adventure = st.slider("🧗 Adventure", 1, 10, 5)
            family = st.slider("👨‍👩‍👧 Family", 1, 10, 5)
            safety = st.slider("🛡️ Safety", 1, 10, 8)
        
        st.markdown("---")
        get_recs_btn = st.button("🔍 Get Recommendations", type="primary", width="stretch")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 **Recommendations**", 
        "💬 **AI Chatbot**",
        "📊 **Explorer**", 
        "📈 **Analytics**",
        "ℹ️ **About**"
    ])
    
    # TAB 1: Recommendations
    with tab1:
        if get_recs_btn:
            with st.spinner("🔮 Finding your perfect destinations..."):
                user_prefs = {
                    'type': destination_type,
                    'budget': budget,
                    'temperature': temperature,
                    'nightlife': nightlife,
                    'culture': culture,
                    'nature': nature,
                    'adventure': adventure,
                    'family_friendly': family,
                    'safety': safety
                }
                
                recommendations = get_ml_recommendations(user_prefs, model_artifacts, top_n=5)
                
                st.success("✅ Found your perfect matches!")
                
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        st.markdown(f"### 🏆 #{i} {rec['city']}, {rec['country']}")
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col1:
                            st.metric("🎯 Match Score", f"{rec['similarity_score']:.1f}%")
                            st.caption(f"**Type:** {rec['type']}")
                            st.caption(f"**Budget:** {rec['cost_level']}")
                        
                        with col2:
                            st.markdown(f"""
                            **🌡️ Temperature:** {rec['avg_temp']}°C | **⭐ Rating:** {rec['tourist_rating']}/5
                            
                            **🛡️ Safety:** {rec['safety_rating']}/5 | **🍽️ Food:** {rec['food_rating']}/5
                            
                            **📅 Best Season:** {rec['best_season']} | **⏱️ Avg Stay:** {rec['avg_stay_days']} days
                            """)
                        
                        with col3:
                            st.markdown("**🌟 Best For:**")
                            if rec['nightlife_score'] >= 7:
                                st.markdown("🌃 Great Nightlife")
                            if rec['culture_score'] >= 7:
                                st.markdown("🎭 Rich Culture")
                            if rec['adventure_score'] >= 7:
                                st.markdown("🧗 Adventure Hub")
                            if rec['nature_score'] >= 7:
                                st.markdown("🌲 Nature Paradise")
                        
                        st.markdown("---")
                
                # Gemini AI Insights
                st.markdown("### 🤖 AI Travel Insights")
                ai_insights = get_gemini_insights(user_prefs, recommendations)
                st.info(ai_insights)
        else:
            st.info("👈 **Set your preferences in the sidebar and click 'Get Recommendations'**")
            
            st.markdown("### 🌍 Featured Destinations")
            sample = df.sample(min(6, len(df)))
            
            cols = st.columns(3)
            for idx, (_, row) in enumerate(sample.iterrows()):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="destination-card">
                        <h3>{row['city']}</h3>
                        <p>📍 {row['country']} | {row['type']}</p>
                        <p>⭐ {row['tourist_rating']}/5 | 💰 {row['cost_level']}</p>
                        <p>🌡️ {row['avg_temp']}°C | 📅 {row['best_season']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # TAB 2: AI Chatbot - FIXED WITH DARK TEXT
    with tab2:
        st.markdown("### 💬 Travel AI Chatbot")
        
        if gemini_model:
            st.success("✅ AI Chatbot is ready! Ask me anything about travel destinations.")
            
            # Display chat history with VISIBLE DARK TEXT
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong style="color: #0D47A1;">🧑 You:</strong><br>
                        <span style="color: #1565C0;">{message['content']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong style="color: #1B5E20;">🤖 Travel AI:</strong><br>
                        <span style="color: #33691E;">{message['content']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Input area
            col1, col2 = st.columns([5, 1])
            with col1:
                user_input = st.text_input(
                    "Type your message:",
                    placeholder="e.g., What are the best beach destinations for families?",
                    key="chat_input",
                    label_visibility="collapsed"
                )
            with col2:
                send_btn = st.button("Send 🚀", width="stretch")
            
            # Suggested questions
            st.markdown("**💡 Suggested Questions:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🏖️ Best beaches?"):
                    user_input = "What are the best beach destinations?"
                    send_btn = True
            with col2:
                if st.button("💰 Budget travel?"):
                    user_input = "Which destinations are best for budget travelers?"
                    send_btn = True
            with col3:
                if st.button("🎭 Cultural cities?"):
                    user_input = "Which cities have the richest culture?"
                    send_btn = True
            
            if send_btn and user_input:
                st.session_state.chat_history.append({'role': 'user', 'content': user_input})
                
                with st.spinner("🤔 Thinking..."):
                    bot_response = chat_with_gemini(user_input, df)
                
                st.session_state.chat_history.append({'role': 'bot', 'content': bot_response})
                st.rerun()

            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

        else:
            st.warning("🔑 **Gemini AI not configured.**")
            st.info("AI model initialization failed. Check console for details.")
    
    # TAB 3: Dataset Explorer
    with tab3:
        st.markdown("### 📊 Explore Travel Destinations")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            type_filter = st.multiselect("Filter by Type", df['type'].unique(), default=list(df['type'].unique()))
        with col2:
            budget_filter = st.multiselect("Filter by Budget", df['cost_level'].unique(), default=list(df['cost_level'].unique()))
        with col3:
            min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
        
        filtered_df = df[
            (df['type'].isin(type_filter)) & 
            (df['cost_level'].isin(budget_filter)) & 
            (df['tourist_rating'] >= min_rating)
        ]
        
        st.dataframe(filtered_df, width="stretch", height=400)
        
        st.markdown("### 📈 Quick Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Destinations", len(filtered_df))
        with col2:
            st.metric("Avg Rating", f"{filtered_df['tourist_rating'].mean():.2f}/5")
        with col3:
            st.metric("Avg Safety", f"{filtered_df['safety_rating'].mean():.2f}/5")
        with col4:
            st.metric("Avg Stay", f"{filtered_df['avg_stay_days'].mean():.1f} days")
    
    # TAB 4: Analytics
    with tab4:
        st.markdown("### 📈 Travel Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Destinations by Type")
            type_counts = df['type'].value_counts()
            fig1 = px.pie(values=type_counts.values, names=type_counts.index, hole=0.4)
            st.plotly_chart(fig1, width="stretch")
        
        with col2:
            st.markdown("#### Budget Distribution")
            budget_counts = df['cost_level'].value_counts()
            fig2 = px.bar(x=budget_counts.index, y=budget_counts.values)
            st.plotly_chart(fig2, width="stretch")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### 🏆 Top Rated")
            top_rated = df.nlargest(10, 'tourist_rating')[['city', 'country', 'tourist_rating']]
            st.dataframe(top_rated, width="stretch", hide_index=True)
        
        with col4:
            st.markdown("#### 🛡️ Safest")
            safest = df.nlargest(10, 'safety_rating')[['city', 'country', 'safety_rating']]
            st.dataframe(safest, width="stretch", hide_index=True)
    
    # TAB 5: About
    with tab5:
        st.markdown("### ℹ️ About This Project")
        st.markdown("""
        ## 🎯 AI Travel Recommender System
        
        ### 🔧 Technologies:
        - **ML:** K-NN + Random Forest
        - **AI:** Google Gemini 1.5
        - **Frontend:** Streamlit
        - **Data:** Pandas, NumPy, Scikit-learn
        
        ### ✨ Features:
        1. Smart ML Recommendations
        2. AI Chatbot Assistant
        3. Dataset Explorer (61 destinations)
        4. Analytics Dashboard
        5. Personalized Travel Insights
        
        ### 📊 Dataset:
        - 61 destinations worldwide
        - 16 features per destination
        - Categories: Beach, Cultural, Urban, Historical, Nature, Adventure
        
        
        """)


if __name__ == "__main__":
    main()