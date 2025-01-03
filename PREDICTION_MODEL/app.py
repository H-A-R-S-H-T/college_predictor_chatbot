import streamlit as st
from college_predictor import CollegePredictor
import plotly.graph_objects as go
import pandas as pd

class CollegePredictorChatbot:
    def __init__(self):
        self.predictor = CollegePredictor()
        self.predictor.load_and_combine_data()
        self.predictor.train_models()
        
    def initialize_session_state(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 'welcome'
        if 'user_inputs' not in st.session_state:
            st.session_state.user_inputs = {}

    def display_chat_interface(self):
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    def add_message(self, role, content):
        st.session_state.messages.append({"role": role, "content": content})

    def handle_welcome(self):
        welcome_message = """
        👋 Welcome to the College Predictor Chatbot! I can help you:
        
        1. 🎯 Get college predictions based on your rank
        2. 📊 View model performance metrics
        3. 📈 Explore training visualizations
        
        What would you like to do? (Enter 1, 2, or 3)
        """
        self.add_message("assistant", welcome_message)
        return "awaiting_initial_choice"

    def create_prediction_visualization(self, predictions):
        if predictions is None or predictions.empty:
            return None
            
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=predictions['College'],
            y=predictions['Cutoff_Rank'],
            text=predictions['Cutoff_Rank'].round(0),
            textposition='auto',
            marker=dict(
                color=predictions.index,
                colorscale='Viridis',
            ),
            hovertemplate='<b>%{x}</b><br>' +
                         'Cutoff Rank: %{y}<br>' +
                         'Branch: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=predictions['Branch']
        ))
        
        fig.update_layout(
            title='Top College Recommendations',
            xaxis_title='College',
            yaxis_title='Cutoff Rank',
            template='plotly_white',
            height=500
        )
        
        return fig

    def run(self):
        st.title("🎓 College Predictor Chatbot")
        self.initialize_session_state()
        
        # Initialize chat if it's the first time
        if not st.session_state.messages:
            st.session_state.current_step = self.handle_welcome()
            
        # Display chat interface
        self.display_chat_interface()
        
        # Get user input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to chat
            self.add_message("user", user_input)
            
            # Handle different conversation states
            if st.session_state.current_step == "awaiting_initial_choice":
                self.handle_initial_choice(user_input)
            elif st.session_state.current_step == "awaiting_rank":
                self.handle_rank_input(user_input)
            elif st.session_state.current_step == "awaiting_category":
                self.handle_category_input(user_input)
            elif st.session_state.current_step == "awaiting_branch":
                self.handle_branch_input(user_input)
            elif st.session_state.current_step == "awaiting_location":
                self.handle_location_input(user_input)

    def handle_initial_choice(self, choice):
        if choice == "1":
            self.add_message("assistant", "Please enter your rank:")
            st.session_state.current_step = "awaiting_rank"
        elif choice == "2":
            # Show model performance
            performance_metrics = self.predictor.get_model_performance()
            self.add_message("assistant", f"📊 Model Performance:\n\n{performance_metrics}")
            st.session_state.current_step = "welcome"
        elif choice == "3":
            # Show available categories
            categories = ", ".join(self.predictor.rank_columns)
            self.add_message("assistant", f"Available categories:\n{categories}\n\nWhich category would you like to see?")
            st.session_state.current_step = "awaiting_category"
        else:
            self.add_message("assistant", "❌ Invalid choice. Please enter 1, 2, or 3.")

    # ... Additional handler methods for rank, category, branch, and location inputs ... 