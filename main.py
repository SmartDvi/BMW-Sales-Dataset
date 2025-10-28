import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, Input, Output, callback, _dash_renderer
import dash_mantine_components as dmc


_dash_renderer._set_react_version("18.2.0")

# reading the dataset frim the file 
df = pd.read_csv("https://github.com/SmartDvi/BMW-Sales-Dataset/blob/main/BMW%20sales%20data%20(2010-2024)%20(1).csv")
print(df)
# preparing and deveing more feature for better predictions
df['Vehicle_Age'] = 2025 - df['Year']
df['Price_Per_KM'] = df['Price_USD'] / (df['Mileage_KM'] + 1)
df['Engine_Price_Ratio'] = df['Price_USD'] / df['Engine_Size_L']

# Encode categorical variable 
label_encoders = {}
categorical_columns = ['Model', 'Region', 'Color', 'Fuel_Type', 'Transmission', 'Sales_Classification']

for col in categorical_columns:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le


class Business_Improvement_Solution:
    def __init__(self, df):
        self.df = df
        self.models = {}
        self.scalers = {}

    def prepare_feature(self, target):
        feature_columns = [
            'Year', 'Engine_Size_L', 'Mileage_KM', 'Vehicle_Age',
            'Price_Per_KM', 'Engine_Price_Ratio',
            'Model_encoded', 'Region_encoded', 'Color_encoded', 
            'Fuel_Type_encoded', 'Transmission_encoded'
        ]

        X = self.df[feature_columns]
        y = self.df[target]

        return train_test_split(X,y, test_size=0.2, random_state=42)
    
    def train_price_prediction_model(self):
        X_train, X_test, y_train, y_test = self.prepare_feature('Price_USD')

        # Scsle the features in preparation for the regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['price'] = scaler

        # training multiple model for this analysis
        models = {
            'Random Forest' : RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression' : LinearRegression(),
            'SVR' : SVR(kernel='rbf')
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
        
        self.models['Price_Prediction'] = best_model

        return best_score
    


    def train_sales_classification_model(self):
        X_train, X_test, y_train, y_test = self.prepare_feature("Sales_Classification_encoded")

        models = {
            'Random Forest' : RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression' : LogisticRegression(random_state=42),
            'SVM' : SVC(random_state=42)
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
        
        self.models['sales_classification'] = best_model

        return best_score
    


    def train_sales_volume_prediction(self):
        X_train, X_test, y_train, y_test = self.prepare_feature("Sales_Volume")

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        self.models['sales_volume'] = model

        return mae
    

    def train_profitability_clustering(self):
        features = ['Price_USD', 'Sales_Volume', 'Vehicle_Age', 'Engine_Size_L']
        X = self.df[features]

        # scaling the features in preparation for the prediction
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['clustering'] = scaler

        # find optional number of clusters
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        # appyling the elbow to determine te clusters 
        optimal_clusters = 4
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        self.df['Profitability_cluster'] = kmeans.fit_predict(X_scaled)
        self.models['clustering'] = kmeans

        return optimal_clusters
    

    def train_region_preference_model(self):
        # Analyze which models/features are preferred in which regions
        region_preferences = self.df.groupby(['Region', 'Model']).agg({
            'Sales_Volume': 'mean',
            'Price_USD': 'mean'
        }).reset_index()
        
        self.models['region_preferences'] = region_preferences
        return region_preferences
    
    def train_fuel_type_trend_model(self):
        fuel_trends = self.df.groupby(['Year', 'Fuel_Type']).agg({
            'Sales_Volume': 'sum',
            'Price_USD': 'mean'
        }).reset_index()
        
        self.models['fuel_trends'] = fuel_trends
        return fuel_trends
    


# Initialize and train all models
ml_solutions = Business_Improvement_Solution(df)
print("Training Machine Learning Models for Business Solutions...")

price_mae = ml_solutions.train_price_prediction_model()
classification_accuracy = ml_solutions.train_sales_classification_model()
volume_mae = ml_solutions.train_sales_volume_prediction()
clusters = ml_solutions.train_profitability_clustering()
region_prefs = ml_solutions.train_region_preference_model()
fuel_trends = ml_solutions.train_fuel_type_trend_model()


app = Dash(__name__, external_stylesheets=[dmc.Styles.ALL])


app.layout = html.Div([
    html.H1("BMW Sales Analytics Dashboard"),
    html.P("Machine learning models trained successfully!"),
    html.P(f"Best Price Prediction Model MAE: ${price_mae:.2f}"),
    html.P(f"Sales Classification Accuracy: {classification_accuracy:.2%}")
])










if __name__ == '__main__':
    app.run(debug=True, port=6040)


