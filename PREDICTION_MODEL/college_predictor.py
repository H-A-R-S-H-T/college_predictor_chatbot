import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_curve, roc_auc_score)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import warnings
import joblib
import os
import time
from tqdm import tqdm
from termcolor import colored
warnings.filterwarnings('ignore')

class CollegePredictor:
    def __init__(self):
        print("Initializing College Predictor...")
        print("\nAvailable datasets:")
        self.datasets = {
            '2019': "C:\\Users\\LENOVO\\Downloads\\CET_Database_Final2019.csv",
            '2020': "C:\\Users\\LENOVO\\Downloads\\CET_Database_Final2020 (1).csv"
        }
        
        # Print available datasets
        for year, path in self.datasets.items():
            print(f"Year {year}: {path}")
            
        self.models = {}
        self.data = None
        self.categorical_features = ['Branch', 'Location']
        self.models_dir = 'trained_models'
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        print("\nInitialization complete!")

    def load_and_combine_data(self):
        """Load and combine data by averaging ranks across years"""
        try:
            all_data = []
            print("\nLoading and combining datasets...")
            
            for year, path in self.datasets.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Dataset file not found: {path}")
                print(f"Loading {year} dataset...")
                df = pd.read_csv(path)
                df['Year'] = year
                all_data.append(df)
            
            # Combine all datasets
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Define rank columns
            self.rank_columns = [col for col in combined_df.columns if any(x in col.upper() for x in 
                ['1G', '1K', '1R', '2AG', '2AK', '2AR', '2BG', '2BK', '2BR',
                 '3AG', '3AK', '3AR', '3BG', '3BK', '3BR', 'GM', 'GMK', 'GMR',
                 'SCG', 'SCK', 'SCR', 'STG', 'STK', 'STR'])]
            
            print("\nProcessing rank data...")
            # Convert ranks to numeric and handle missing values
            for col in self.rank_columns:
                combined_df[col] = pd.to_numeric(
                    combined_df[col].astype(str).str.replace(',', ''), 
                    errors='coerce'
                )
            
            # Process categorical features
            print("Processing categorical data...")
            for col in self.categorical_features:
                combined_df[col] = combined_df[col].fillna('Not Specified')
                combined_df[col] = combined_df[col].replace('', 'Not Specified')
                combined_df[col] = combined_df[col].str.strip()
            
            # Group by college attributes and calculate mean ranks
            print("Calculating average ranks across years...")
            self.data = (combined_df.groupby(['College', 'Branch', 'Location', 'CETCode'], 
                                           as_index=False)
                        .agg({col: 'mean' for col in self.rank_columns}))
            
            # Fill any remaining NaN values with median
            for col in self.rank_columns:
                self.data[col] = self.data[col].fillna(self.data[col].median())
            
            print(f"\n✓ Successfully processed {len(self.data)} unique college-branch combinations")
            print(f"✓ Using {len(self.rank_columns)} rank categories")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error loading data: {str(e)}")
            return False

    def create_analysis_plots(self, category, X, y, model, plots_dir):
        """Create and save analysis plots with all metrics"""
        try:
            # Data Distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(data=self.data, x=category, bins=50)
            plt.title(f'Combined Rank Distribution for {category}')
            plt.xlabel('Rank')
            plt.ylabel('Count')
            plt.savefig(os.path.join(plots_dir, 'rank_distribution.png'))
            plt.close()

            # Branch Distribution
            plt.figure(figsize=(15, 6))
            sns.countplot(data=self.data, x='Branch')
            plt.xticks(rotation=45, ha='right')
            plt.title('Branch Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'branch_distribution.png'))
            plt.close()

            # Model evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Fit model on training data
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)

            # Save metrics report
            with open(os.path.join(plots_dir, 'metrics.txt'), 'w') as f:
                f.write(f"Unified Model Metrics for {category}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Accuracy Score: {accuracy:.4f}\n")
                f.write(f"ROC AUC Score: {auc_score:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(class_report)

            # Confusion Matrix Plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
            plt.close()

            # ROC Curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(plots_dir, 'roc_curve.png'))
            plt.close()

            return {
                'accuracy': accuracy,
                'roc_auc': auc_score,
                'classification_report': class_report
            }

        except Exception as e:
            print(f"Error creating plots: {str(e)}")
            return None

    def create_model(self):
        """Create and return the XGBoost model with preprocessing pipeline"""
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['Rank']),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, 
                                    handle_unknown='ignore'), 
                 ['Branch', 'Location'])
            ])
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            ))
        ])
        
        return model

    def train_models(self, retrain=False):
        """Train a single unified model for all categories with visualizations"""
        try:
            model_file = os.path.join(self.models_dir, 'unified_model.joblib')
            label_encoder_file = os.path.join(self.models_dir, 'label_encoder.joblib')
            plots_dir = os.path.join(self.models_dir, 'training_plots')
            
            # Create plots directory if it doesn't exist
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            
            # Load existing model if available and not retraining
            if os.path.exists(model_file) and not retrain:
                print("✓ Loading existing unified model")
                self.unified_model = joblib.load(model_file)
                self.label_encoder = joblib.load(label_encoder_file)
                return True
            
            print("\n🔄 Training new unified model...")
            print("="*50)
            
            # Data preparation with progress bar
            print("\nPreparing training data...")
            train_data = []
            for category in tqdm(self.rank_columns, desc="Processing categories"):
                category_data = pd.DataFrame({
                    'Rank': self.data[category],
                    'Branch': self.data['Branch'],
                    'Location': self.data['Location'],
                    'Category': category,
                    'College': self.data['College'].str.strip(),
                    'CETCode': self.data['CETCode']
                })
                train_data.append(category_data)
            
            # Combine and process data
            training_df = pd.concat(train_data, ignore_index=True)
            training_df = training_df.dropna()
            
            # Create initial visualizations
            print("\n📊 Creating data analysis plots...")
            
            # 1. Distribution of ranks
            plt.figure(figsize=(12, 6))
            sns.histplot(data=training_df, x='Rank', bins=50)
            plt.title('Distribution of Ranks Across All Categories')
            plt.xlabel('Rank')
            plt.ylabel('Count')
            plt.savefig(os.path.join(plots_dir, 'rank_distribution.png'))
            plt.close()
            
            # 2. Category distribution
            plt.figure(figsize=(15, 6))
            sns.countplot(data=training_df, x='Category')
            plt.title('Distribution of Categories')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'category_distribution.png'))
            plt.close()
            
            # 3. Branch distribution
            plt.figure(figsize=(15, 6))
            branch_counts = training_df['Branch'].value_counts()
            plt.bar(range(len(branch_counts)), branch_counts.values)
            plt.title('Distribution of Branches')
            plt.xlabel('Branch')
            plt.ylabel('Count')
            plt.xticks(range(len(branch_counts)), branch_counts.index, rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'branch_distribution.png'))
            plt.close()
            
            # Prepare features
            print("\nPreparing features...")
            X = training_df[['Rank', 'Branch', 'Location', 'Category']]
            y = training_df['College']
            
            # Convert labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Create and train model
            print("\n🚀 Training model...")
            xgb_params = {
                'n_estimators': 300,
                'learning_rate': 0.1,
                'max_depth': 7,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss',
                'objective': 'multi:softprob',
                'num_class': len(self.label_encoder.classes_),
                'verbosity': 0
            }
            
            model = Pipeline([
                ('preprocessor', ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), ['Rank']),
                        ('cat', OneHotEncoder(drop='first', sparse_output=False, 
                                            handle_unknown='ignore'), 
                         ['Branch', 'Location', 'Category'])
                    ])),
                ('classifier', xgb.XGBClassifier(**xgb_params))
            ])
            
            # Train with progress tracking
            with tqdm(total=100, desc="Training progress") as pbar:
                model.fit(X_train, y_train)
                pbar.update(100)
            
            # Evaluate model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Create performance visualization
            plt.figure(figsize=(10, 6))
            scores = [train_score, test_score]
            plt.bar(['Training', 'Testing'], scores, color=['#2ecc71', '#3498db'])
            plt.title('Model Performance')
            plt.ylabel('Accuracy Score')
            for i, v in enumerate(scores):
                plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
            plt.ylim(0, 1.0)
            plt.savefig(os.path.join(plots_dir, 'model_performance.png'))
            plt.close()
            
            # Save model and encoder
            print("\n💾 Saving model and artifacts...")
            joblib.dump(model, model_file)
            joblib.dump(self.label_encoder, label_encoder_file)
            self.unified_model = model
            
            # Print performance summary
            print("\n📈 Model Performance Summary")
            print("="*50)
            print(f"Training Accuracy: {train_score:.3f}")
            print(f"Testing Accuracy:  {test_score:.3f}")
            print("\n✨ Training visualizations saved in:", plots_dir)
            print("="*50)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error training unified model: {str(e)}")
            return False

    def process_predictions(self, colleges, category, rank, branch, location):
        """Process and filter college predictions with interactive location selection"""
        try:
            # Filter eligible colleges
            eligible_colleges = colleges[colleges[category] >= float(rank)].copy()
            
            # Location filtering with interactive selection
            if location and location.lower() != 'skip':
                location = location.lower().strip()
                
                # Get all unique locations
                all_locations = eligible_colleges['Location'].unique()
                
                # Function to calculate similarity score
                def calculate_similarity(college_location):
                    college_location = college_location.lower()
                    # Split into words and check for partial matches
                    input_words = set(location.split())
                    college_words = set(college_location.split())
                    
                    # Check for partial word matches
                    matching_score = 0
                    for input_word in input_words:
                        for college_word in college_words:
                            if input_word in college_word or college_word in input_word:
                                matching_score += 1
                    return matching_score
                
                # Calculate similarity scores
                location_scores = [(loc, calculate_similarity(loc)) for loc in all_locations]
                
                # Filter locations with any match
                matching_locations = [loc for loc, score in location_scores if score > 0]
                
                if matching_locations:
                    print("\n📍 Found these matching locations:")
                    for i, loc in enumerate(matching_locations, 1):
                        print(f"{i}. {loc}")
                    
                    while True:
                        choice = input("\nEnter location number (or 0 to see all locations): ")
                        if choice == '0':
                            print("\n📍 All available locations:")
                            for i, loc in enumerate(sorted(all_locations), 1):
                                print(f"{i}. {loc}")
                            choice = input("\nEnter location number: ")
                            
                        if choice.isdigit():
                            choice = int(choice)
                            if choice == 0:
                                return None
                            elif 0 < choice <= len(all_locations):
                                selected_location = (matching_locations if choice <= len(matching_locations) 
                                                  else sorted(all_locations))[choice-1]
                                eligible_colleges = eligible_colleges[
                                    eligible_colleges['Location'] == selected_location]
                                print(f"\n✓ Selected: {selected_location}")
                                break
                        print("❌ Invalid choice. Please try again.")
                else:
                    print(f"\n❌ No locations found matching '{location}'")
                    print("\n📍 Available locations:")
                    for i, loc in enumerate(sorted(all_locations), 1):
                        print(f"{i}. {loc}")
                    
                    choice = input("\nEnter location number: ")
                    if choice.isdigit() and 0 < int(choice) <= len(all_locations):
                        selected_location = sorted(all_locations)[int(choice)-1]
                        eligible_colleges = eligible_colleges[
                            eligible_colleges['Location'] == selected_location]
                        print(f"\n✓ Selected: {selected_location}")
                    else:
                        return None
            
            # Simple branch filtering
            if branch and branch.lower() != 'skip':
                branch = branch.lower().strip()
                eligible_colleges['Branch_Lower'] = eligible_colleges['Branch'].str.lower()
                branch_matches = eligible_colleges[eligible_colleges['Branch_Lower'].str.contains(branch, na=False)]
                
                if not branch_matches.empty:
                    eligible_colleges = branch_matches
                else:
                    print(f"\n❌ No colleges found with branch '{branch}' in selected location")
                    return None
                
                eligible_colleges = eligible_colleges.drop('Branch_Lower', axis=1)
            
            # Sort by rank (ascending = better ranks first)
            result = (eligible_colleges
                     .sort_values(category, ascending=True)
                     .drop_duplicates(['College', 'Branch', 'Location'])
                     .head(10)
                     [['College', 'Branch', 'Location', category, 'CETCode']])
            
            if result.empty:
                print("\n❌ No colleges found matching all criteria.")
                return None
                
            return result.rename(columns={category: 'Cutoff_Rank'})
            
        except Exception as e:
            print(f"Error in processing predictions: {str(e)}")
            return None

    def format_predictions(self, predictions):
        """Format predictions into a readable message"""
        if predictions is None:
            return "❌ Sorry, no colleges found matching your criteria."
        
        message = "🎯 Top College Recommendations (Best to Least Preferred):\n\n"
        
        for i, (_, row) in enumerate(predictions.iterrows(), 1):
            message += f"Rank #{i}\n"
            message += f"🏫 {row['College']}\n"
            message += f"📚 Branch: {row['Branch']}\n"
            message += f"📍 Location: {row['Location']}\n"
            message += f"🎯 CET Code: {row['CETCode']}\n"
            message += f"📊 Cutoff Rank: {row['Cutoff_Rank']:.0f}\n"
            message += "-" * 50 + "\n"
        
        return message

    def predict_colleges(self, rank, category, branch=None, location=None):
        """Predict colleges using the trained unified model"""
        try:
            # Check if unified model exists
            if not hasattr(self, 'unified_model'):
                print("\n⚠️ Model not trained. Training now...")
                if not self.train_models():
                    raise ValueError("Failed to train model")
            
            # Process input
            input_data = pd.DataFrame({
                'Rank': [float(rank)],
                'Branch': [branch if branch else 'Not Specified'],
                'Location': [location if location else 'Not Specified'],
                'Category': [category]
            })
            
            # Get colleges
            colleges = self.data.drop_duplicates(['College', 'Branch', 'Location', category])
            
            # Process results
            result = self.process_predictions(colleges, category, rank, branch, location)
            
            if result is not None and not result.empty:
                return result
            
            return None
            
        except Exception as e:
            print(f"Error in predictions: {str(e)}")
            return None

    def create_enhanced_visualization(self, result, category):
        """Create an enhanced interactive visualization"""
        fig = go.Figure()
        
        # Add bars with gradient colors
        fig.add_trace(go.Bar(
            x=result['College'],
            y=result['Probability'],
            text=[f"{p:.1%}" for p in result['Probability']],
            textposition='auto',
            marker=dict(
                color=result['Probability'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Admission Probability")
            ),
            hovertemplate='<b>%{x}</b><br>' +
                        'Probability: %{text}<br>' +
                        'Cutoff Rank: %{customdata}<br>' +
                        'Branch: %{meta}<extra></extra>',
            customdata=result['Cutoff_Rank'].round(2),
            meta=result['Branch']
        ))
        
        # Update layout with better styling
        fig.update_layout(
            title={
                'text': 'Top 10 Predicted Colleges',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24, color='#2E4053')
            },
            xaxis_title={'text': 'College', 'font': dict(size=14)},
            yaxis_title={'text': 'Admission Probability', 'font': dict(size=14)},
            yaxis_tickformat='.0%',
            template='plotly_white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
            ),
            height=600,
            margin=dict(t=100, b=100),
            plot_bgcolor='rgba(240,240,240,0.95)',
            paper_bgcolor='white'
        )
        
        # Save visualization
        plots_dir = os.path.join(self.models_dir, 'predictions', category)
        os.makedirs(plots_dir, exist_ok=True)
        fig.write_html(os.path.join(plots_dir, 'predictions.html'))

    def run(self):
        """Main loop for chatbot interaction"""
        print("\n" + "="*50)
        print("🎓 Welcome to College Predictor Chatbot!")
        print("="*50)
        
        while True:
            if self.current_state == "INIT":
                print("\nI can help you with:")
                print("1. 🎯 Get college predictions")
                print("2. 📊 Check model performance")
                print("3. 📈 View training visualizations")
                print("4. 🔄 Retrain models")
                print("5. 🚪 Exit")
                
                choice = input("\nWhat would you like to do? (1-5): ")
                
                if choice == "1":
                    self.current_state = "AWAITING_RANK"
                    print("\nPlease enter your rank:")
                elif choice == "2":
                    print("\n" + self.get_model_performance())
                elif choice == "3":
                    categories = ", ".join(self.rank_columns)
                    print(f"\nAvailable categories: {categories}")
                    category = input("Which category would you like to see? ").upper()
                    if category in self.rank_columns:
                        self.create_training_visualization(category)
                        print(f"\n✓ Visualization saved as 'training_viz_{category}.png'")
                    else:
                        print(f"\n❌ Invalid category. Please choose from: {categories}")
                elif choice == "4":
                    self.retrain_models()
                elif choice == "5":
                    print("\nThank you for using College Predictor! Goodbye! 👋")
                    break
                else:
                    print("\n❌ Invalid choice. Please try again.")

def main():
    try:
        print("Starting College Prediction System...")
        
        # Initialize predictor
        predictor = CollegePredictor()
        
        # Load and combine data
        data = predictor.load_and_combine_data()
        if data is None:
            raise ValueError("Failed to load data!")
        
        while True:
            print("\nOptions:")
            print("1. Train/Retrain unified model")
            print("2. Make predictions")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                retrain = input("\nRetrain existing model? (yes/no): ").lower() == 'yes'
                predictor.train_models(retrain=retrain)
                
            elif choice == '2':
                # Display available categories
                print("\nAvailable Categories:")
                for i, cat in enumerate(predictor.rank_columns, 1):
                    print(f"{i}. {cat}")
                
                category = input("\nEnter category (e.g., GM, 1G, 2AG): ")
                if category not in predictor.rank_columns:
                    print(f"Invalid category!")
                    continue
                
                try:
                    rank = float(input("Enter rank: "))
                except ValueError:
                    print("Invalid rank. Please enter a number.")
                    continue
                
                branch = input("Enter preferred branch (or press Enter to skip): ")
                location = input("Enter preferred location (or press Enter to skip): ")
                
                # Make predictions
                predictions = predictor.predict_colleges(rank, category, branch, location)
                
                if predictions is not None:
                    print("\nTop College Recommendations:")
                    print(predictions.to_string(index=False))
                
            elif choice == '3':
                print("\nThank you for using the College Prediction System!")
                break
                
            else:
                print("\nInvalid choice. Please enter 1, 2, or 3.")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()

