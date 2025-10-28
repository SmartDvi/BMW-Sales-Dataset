import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, accuracy_score
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load and prepare data
def load_data():
    df = pd.read_csv("BMW sales data (2010-2024) (1).csv")
    return df

def prepare_data(df):
    """Feature engineering and preprocessing"""
    current_year = datetime.now().year
    df['Vehicle_Age'] = current_year - df['Year']
    df['Price_Per_KM'] = df['Price_USD'] / (df['Mileage_KM'] + 1)
    df['Engine_Price_Ratio'] = df['Price_USD'] / df['Engine_Size_L']
    df['Revenue'] = df['Price_USD'] * df['Sales_Volume']
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Model', 'Region', 'Color', 'Fuel_Type', 'Transmission', 'Sales_Classification']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

# Color scheme
colors = {
    'background': '#0f1a2f',
    'card_bg': '#1e2a45',
    'card_border': '#2d3b55',
    'primary': '#2563eb',
    'secondary': '#06b6d4',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'text_primary': '#f8fafc',
    'text_secondary': '#94a3b8',
    'accent_gradient': 'linear-gradient(135deg, #2563eb 0%, #06b6d4 100%)'
}

# Filter function
def filter_data(df, year_range, selected_years, classification, regions, fuels, models, transmissions, colors_list):
    """Helper function to filter data based on all criteria"""
    filtered_df = df.copy()
    
    # Apply year range filter
    if year_range:
        filtered_df = filtered_df[
            (filtered_df['Year'] >= year_range[0]) & 
            (filtered_df['Year'] <= year_range[1])
        ]
    
    # Apply specific year selection
    if selected_years:
        filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
    
    # Apply other filters
    if classification:
        filtered_df = filtered_df[filtered_df['Sales_Classification'].isin(classification)]
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    if fuels:
        filtered_df = filtered_df[filtered_df['Fuel_Type'].isin(fuels)]
    if models:
        filtered_df = filtered_df[filtered_df['Model'].isin(models)]
    if transmissions:
        filtered_df = filtered_df[filtered_df['Transmission'].isin(transmissions)]
    if colors_list:
        filtered_df = filtered_df[filtered_df['Color'].isin(colors_list)]
    
    return filtered_df

# ML Models Class
class BusinessMLSolutions:
    def __init__(self, df):
        self.df = df
        self.models = {}
        self.scalers = {}
        
    def prepare_features(self, target):
        """Prepare features for different prediction tasks"""
        feature_columns = [
            'Year', 'Engine_Size_L', 'Mileage_KM', 'Vehicle_Age', 
            'Price_Per_KM', 'Engine_Price_Ratio',
            'Model_encoded', 'Region_encoded', 'Color_encoded', 
            'Fuel_Type_encoded', 'Transmission_encoded'
        ]
        
        X = self.df[feature_columns]
        y = self.df[target]
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_price_prediction_model(self):
        """Model 1: Price Prediction for Pricing Optimization"""
        X_train, X_test, y_train, y_test = self.prepare_features('Price_USD')
        
        # Scale features for regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['price'] = scaler
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(kernel='rbf')
        }
        
        best_model = None
        best_score = float('inf')
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            
            if mae < best_score:
                best_score = mae
                best_model = model
                
            print(f"{name} MAE: ${mae:,.2f}")
            
        self.models['price_prediction'] = best_model
        return best_score
    
    def train_sales_classification_model(self):
        """Model 2: Sales Classification for Inventory Management"""
        X_train, X_test, y_train, y_test = self.prepare_features('Sales_Classification_encoded')
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                
            print(f"{name} Accuracy: {accuracy:.3f}")
            
        self.models['sales_classification'] = best_model
        return best_score
    
    def train_sales_volume_prediction(self):
        """Model 3: Sales Volume Prediction for Demand Forecasting"""
        X_train, X_test, y_train, y_test = self.prepare_features('Sales_Volume')
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        self.models['sales_volume'] = model
        print(f"Sales Volume Prediction MAE: {mae:.2f} units")
        return mae
    
    def train_profitability_clustering(self):
        """Model 4: Customer/Product Segmentation for Targeted Marketing"""
        features = ['Price_USD', 'Sales_Volume', 'Vehicle_Age', 'Engine_Size_L']
        X = self.df[features]
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['clustering'] = scaler
        
        # Find optimal number of clusters
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        
        # Use elbow method to determine clusters (simplified)
        optimal_clusters = 4
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        self.df['Profitability_Cluster'] = kmeans.fit_predict(X_scaled)
        self.models['clustering'] = kmeans
        
        return optimal_clusters
    
    def train_region_preference_model(self):
        """Model 5: Regional Preference Analysis for Market Expansion"""
        # Analyze which models/features are preferred in which regions
        region_preferences = self.df.groupby(['Region', 'Model']).agg({
            'Sales_Volume': 'mean',
            'Price_USD': 'mean'
        }).reset_index()
        
        self.models['region_preferences'] = region_preferences
        return region_preferences
    
    def train_fuel_type_trend_model(self):
        """Model 6: Fuel Type Trend Analysis for Strategic Planning"""
        fuel_trends = self.df.groupby(['Year', 'Fuel_Type']).agg({
            'Sales_Volume': 'sum',
            'Price_USD': 'mean'
        }).reset_index()
        
        self.models['fuel_trends'] = fuel_trends
        return fuel_trends

# Initialize data and models
df = load_data()
df, label_encoders = prepare_data(df)
ml_solutions = BusinessMLSolutions(df)

# Train models
print("Training Machine Learning Models for Business Solutions...")
price_mae = ml_solutions.train_price_prediction_model()
classification_accuracy = ml_solutions.train_sales_classification_model()
volume_mae = ml_solutions.train_sales_volume_prediction()
clusters = ml_solutions.train_profitability_clustering()
region_prefs = ml_solutions.train_region_preference_model()
fuel_trends = ml_solutions.train_fuel_type_trend_model()

# Get unique years for filters
years = sorted(df['Year'].unique())
min_year = min(years)
max_year = max(years)