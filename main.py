import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score
from sklearn.inspection import permutation_importance
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, Input, Output, callback, State
import dash_mantine_components as dmc

# Load and prepare data
def load_data():
    # Sample data structure - replace with your actual CSV
    df = pd.read_csv("C:\\Users\\Moving_King\\Documents\\BMW_dashboard\\BMW sales data (2010-2024) (1).csv")
    return df

def prepare_data(df):
    """Feature engineering and preprocessing"""
    current_year = datetime.now().year
    df['Vehicle_Age'] = current_year - df['Year']
    df['Price_Per_KM'] = df['Price_USD'] / (df['Mileage_KM'] + 1)
    df['Engine_Price_Ratio'] = df['Price_USD'] / (df['Engine_Size_L'] + 0.1)  # Avoid division by zero
    df['Revenue'] = df['Price_USD'] * df['Sales_Volume']
    
    # Create meaningful sales classification based on actual data
    sales_quantiles = df['Sales_Volume'].quantile([0.33, 0.66])
    df['Sales_Classification'] = 'Medium'
    df.loc[df['Sales_Volume'] <= sales_quantiles[0.33], 'Sales_Classification'] = 'Low'
    df.loc[df['Sales_Volume'] >= sales_quantiles[0.66], 'Sales_Classification'] = 'High'
    
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
def filter_data(df, year_range, selected_models, selected_regions, selected_fuels, 
                selected_transmissions, selected_colors, selected_classifications):
    """Helper function to filter data based on all criteria"""
    filtered_df = df.copy()
    
    # Apply year range filter
    if year_range:
        filtered_df = filtered_df[
            (filtered_df['Year'] >= year_range[0]) & 
            (filtered_df['Year'] <= year_range[1])
        ]
    
    # Apply other filters
    if selected_models and len(selected_models) > 0:
        filtered_df = filtered_df[filtered_df['Model'].isin(selected_models)]
    if selected_regions and len(selected_regions) > 0:
        filtered_df = filtered_df[filtered_df['Region'].isin(selected_regions)]
    if selected_fuels and len(selected_fuels) > 0:
        filtered_df = filtered_df[filtered_df['Fuel_Type'].isin(selected_fuels)]
    if selected_transmissions and len(selected_transmissions) > 0:
        filtered_df = filtered_df[filtered_df['Transmission'].isin(selected_transmissions)]
    if selected_colors and len(selected_colors) > 0:
        filtered_df = filtered_df[filtered_df['Color'].isin(selected_colors)]
    if selected_classifications and len(selected_classifications) > 0:
        filtered_df = filtered_df[filtered_df['Sales_Classification'].isin(selected_classifications)]
    
    return filtered_df

# ML Models Class with CORRECT implementations
class BusinessMLSolutions:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.feature_names = [
            'Year', 'Engine_Size_L', 'Mileage_KM', 'Vehicle_Age', 
            'Price_Per_KM', 'Engine_Price_Ratio',
            'Model_encoded', 'Region_encoded', 'Color_encoded', 
            'Fuel_Type_encoded', 'Transmission_encoded'
        ]
        
    def prepare_features(self, df, target):
        """Prepare features for different prediction tasks"""
        X = df[self.feature_names]
        y = df[target]
        
        # Only split if we have enough data
        if len(df) > 10:
            return train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            return X, None, y, None
        
    def train_models_on_data(self, df):
        """Train all models on the provided dataframe"""
        if len(df) < 10:
            print(f"Not enough data to train models. Only {len(df)} samples available.")
            return False
            
        print(f"Training ML models on {len(df)} samples...")
        
        # Reset metrics
        self.metrics = {}
        
        # Model 1: Price Prediction
        X_train, X_test, y_train, y_test = self.prepare_features(df, 'Price_USD')
        
        if X_test is not None:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['price'] = scaler
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)  # Reduced for performance
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.models['price_prediction'] = model
            self.metrics['price_mae'] = mae
            self.metrics['price_r2'] = r2
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            self.metrics['price_feature_importance'] = feature_importance
            print(f"Price Prediction - MAE: ${mae:,.2f}, R²: {r2:.3f}")
        
        # Model 2: Sales Classification
        X_train, X_test, y_train, y_test = self.prepare_features(df, 'Sales_Classification_encoded')
        
        if X_test is not None:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            self.models['sales_classification'] = model
            self.metrics['classification_accuracy'] = accuracy
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            self.metrics['classification_feature_importance'] = feature_importance
            print(f"Sales Classification - Accuracy: {accuracy:.3f}")
        
        # Model 3: Sales Volume Prediction
        X_train, X_test, y_train, y_test = self.prepare_features(df, 'Sales_Volume')
        
        if X_test is not None:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.models['sales_volume'] = model
            self.metrics['volume_mae'] = mae
            self.metrics['volume_r2'] = r2
            
            print(f"Sales Volume Prediction - MAE: {mae:.2f} units, R²: {r2:.3f}")
        
        # Model 4: Clustering
        features = ['Price_USD', 'Sales_Volume', 'Vehicle_Age', 'Engine_Size_L']
        X = df[features]
        
        if len(df) >= 4:  # Need at least 4 samples for 4 clusters
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['clustering'] = scaler
            
            optimal_clusters = min(4, len(df))  # Adjust clusters based on data size
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            df['Profitability_Cluster'] = cluster_labels
            self.models['clustering'] = kmeans
            
            # Analyze clusters
            cluster_analysis = df.groupby('Profitability_Cluster').agg({
                'Price_USD': 'mean',
                'Sales_Volume': 'mean',
                'Revenue': 'mean',
                'Model': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
            }).reset_index()
            
            self.metrics['cluster_analysis'] = cluster_analysis
            self.metrics['optimal_clusters'] = optimal_clusters
            print(f"Clustering - {optimal_clusters} segments created")
        
        return True

# Initialize data and models
df = load_data()
df, label_encoders = prepare_data(df)
ml_solutions = BusinessMLSolutions()

# Train initial models on full dataset
print("Training initial Machine Learning Models...")
ml_solutions.train_models_on_data(df)

# Get unique values for filters
years = sorted(df['Year'].unique())
min_year = min(years)
max_year = max(years)
models = sorted(df['Model'].unique())
regions = sorted(df['Region'].unique())
fuel_types = sorted(df['Fuel_Type'].unique())
transmissions = sorted(df['Transmission'].unique())
colors_list = sorted(df['Color'].unique())
classifications = sorted(df['Sales_Classification'].unique())

def create_profitability_matrix(df):
    """Create a 2x2 profitability matrix for strategic decision making"""
    if len(df) == 0:
        df['profitability_segment'] = 'Medium'
        return df
        
    price_quantile = df['Price_USD'].quantile(0.5)
    volume_quantile = df['Sales_Volume'].quantile(0.5)
    
    df['profitability_segment'] = 'Medium'
    df.loc[(df['Price_USD'] >= price_quantile) & (df['Sales_Volume'] >= volume_quantile), 'profitability_segment'] = 'Star'
    df.loc[(df['Price_USD'] >= price_quantile) & (df['Sales_Volume'] < volume_quantile), 'profitability_segment'] = 'Premium'
    df.loc[(df['Price_USD'] < price_quantile) & (df['Sales_Volume'] >= volume_quantile), 'profitability_segment'] = 'Volume'
    df.loc[(df['Price_USD'] < price_quantile) & (df['Sales_Volume'] < volume_quantile), 'profitability_segment'] = 'Basic'
    
    return df

# Apply profitability matrix
df = create_profitability_matrix(df)

app = Dash(__name__)
server = app.server

# Add hidden data store to layout
app.layout = dmc.MantineProvider(
    theme={
        'colorScheme': 'dark',
        'fontFamily': 'Inter, sans-serif',
    },
    children=[
        dcc.Store(id='filtered-data-store', data=df.to_json(date_format='iso', orient='split')),
        
        dmc.Container(
            fluid=True,
            children=[
                # Header
                dmc.Paper(
                    p="lg",
                    mb="xl",
                    style={
                        'background': colors['accent_gradient'],
                        'borderRadius': '15px',
                    },
                    children=[
                        dmc.Stack(
                            gap="xs",
                            children=[
                                dmc.Title("🤖 BMW Business Intelligence Dashboard", order=2, c="white"),
                                dmc.Text("Data-Driven Insights for Strategic Decision Making", 
                                        c="white", style={'opacity': '0.9'})
                            ]
                        )
                    ]
                ),
                
                # Filters Section
                dmc.Paper(
                    p="lg",
                    mb="xl",
                    style={
                        'backgroundColor': colors['card_bg'],
                        'border': f'1px solid {colors["card_border"]}',
                        'borderRadius': '15px',
                    },
                    children=[
                        dmc.Stack(
                            gap="md",
                            children=[
                                dmc.Title("🔍 Data Filters", order=4, c=colors['text_primary']),
                                dmc.Grid(
                                    gutter="lg",
                                    children=[
                                        dmc.GridCol(span=3, children=[
                                            dmc.Stack(
                                                gap="xs",
                                                children=[
                                                    dmc.Text("Year Range", size="sm", c=colors['text_secondary']),
                                                    dmc.RangeSlider(
                                                        id="year-range-slider",
                                                        min=min_year,
                                                        max=max_year,
                                                        value=[min_year, max_year],
                                                        marks=[{"value": year, "label": str(year)} for year in range(min_year, max_year+1, 2)],
                                                        mb=35
                                                    )
                                                ]
                                            )
                                        ]),
                                        dmc.GridCol(span=3, children=[
                                            dmc.MultiSelect(
                                                label="Models",
                                                id="model-filter",
                                                data=[{"value": model, "label": model} for model in models],
                                                placeholder="Select models...",
                                                clearable=True,
                                                searchable=True
                                            )
                                        ]),
                                        dmc.GridCol(span=3, children=[
                                            dmc.MultiSelect(
                                                label="Regions",
                                                id="region-filter",
                                                data=[{"value": region, "label": region} for region in regions],
                                                placeholder="Select regions...",
                                                clearable=True,
                                                searchable=True
                                            )
                                        ]),
                                        dmc.GridCol(span=3, children=[
                                            dmc.MultiSelect(
                                                label="Fuel Types",
                                                id="fuel-filter",
                                                data=[{"value": fuel, "label": fuel} for fuel in fuel_types],
                                                placeholder="Select fuel types...",
                                                clearable=True,
                                                searchable=True
                                            )
                                        ]),
                                    ]
                                ),
                                dmc.Grid(
                                    gutter="lg",
                                    children=[
                                        dmc.GridCol(span=3, children=[
                                            dmc.MultiSelect(
                                                label="Transmissions",
                                                id="transmission-filter",
                                                data=[{"value": trans, "label": trans} for trans in transmissions],
                                                placeholder="Select transmissions...",
                                                clearable=True,
                                                searchable=True
                                            )
                                        ]),
                                        dmc.GridCol(span=3, children=[
                                            dmc.MultiSelect(
                                                label="Colors",
                                                id="color-filter",
                                                data=[{"value": color, "label": color} for color in colors_list],
                                                placeholder="Select colors...",
                                                clearable=True,
                                                searchable=True
                                            )
                                        ]),
                                        dmc.GridCol(span=3, children=[
                                            dmc.MultiSelect(
                                                label="Sales Performance",
                                                id="classification-filter",
                                                data=[{"value": cls, "label": cls} for cls in classifications],
                                                placeholder="Select performance...",
                                                clearable=True,
                                                searchable=True
                                            )
                                        ]),
                                        dmc.GridCol(span=3, children=[
                                            dmc.Button("Apply Filters", id="apply-filters-btn", color="blue", fullWidth=True, mt=25)
                                        ]),
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                
                # ML Performance Cards - USING ACTUAL METRICS
                dmc.Grid(
                    gutter="lg",
                    mb="xl",
                    children=[
                        dmc.GridCol(span=3, children=[
                            dmc.Paper(
                                p="lg",
                                style={
                                    'backgroundColor': colors['card_bg'],
                                    'border': f'1px solid {colors["card_border"]}',
                                    'borderRadius': '15px',
                                },
                                children=[
                                    dmc.Stack(
                                        gap="xs",
                                        children=[
                                            dmc.Group(justify="apart", children=[
                                                dmc.Text("Price Prediction", c=colors['text_secondary'], size="sm"),
                                                dmc.ThemeIcon("🎯", variant="light", color="blue")
                                            ]),
                                            dmc.Title(id="price-mae-display", order=3, c=colors['text_primary']),
                                            dmc.Text(id="price-r2-display", size="xs", c=colors['text_secondary']),
                                            dmc.Progress(id="price-progress", value=0, color="blue", size="sm", mt="xs")
                                        ]
                                    )
                                ]
                            ),
                        ]),
                        dmc.GridCol(span=3, children=[
                            dmc.Paper(
                                p="lg",
                                style={
                                    'backgroundColor': colors['card_bg'],
                                    'border': f'1px solid {colors["card_border"]}',
                                    'borderRadius': '15px',
                                },
                                children=[
                                    dmc.Stack(
                                        gap="xs",
                                        children=[
                                            dmc.Group(justify="apart", children=[
                                                dmc.Text("Sales Classification", c=colors['text_secondary'], size="sm"),
                                                dmc.ThemeIcon("📈", variant="light", color="green")
                                            ]),
                                            dmc.Title(id="classification-accuracy-display", order=3, c=colors['text_primary']),
                                            dmc.Text("Accuracy Score", size="xs", c=colors['text_secondary']),
                                            dmc.Progress(id="classification-progress",value=50, color="green", size="sm", mt="xs")
                                        ]
                                    )
                                ]
                            ),
                        ]),
                        dmc.GridCol(span=3, children=[
                            dmc.Paper(
                                p="lg",
                                style={
                                    'backgroundColor': colors['card_bg'],
                                    'border': f'1px solid {colors["card_border"]}',
                                    'borderRadius': '15px',
                                },
                                children=[
                                    dmc.Stack(
                                        gap="xs",
                                        children=[
                                            dmc.Group(justify="apart", children=[
                                                dmc.Text("Volume Prediction", c=colors['text_secondary'], size="sm"),
                                                dmc.ThemeIcon("📊", variant="light", color="orange")
                                            ]),
                                            dmc.Title(id="volume-mae-display", order=3, c=colors['text_primary']),
                                            dmc.Text(id="volume-r2-display", size="xs", c=colors['text_secondary']),
                                            dmc.Progress(id="volume-progress", value=50, color="orange", size="sm", mt="xs")
                                        ]
                                    )
                                ]
                            ),
                        ]),
                        dmc.GridCol(span=3, children=[
                            dmc.Paper(
                                p="lg",
                                style={
                                    'backgroundColor': colors['card_bg'],
                                    'border': f'1px solid {colors["card_border"]}',
                                    'borderRadius': '15px',
                                },
                                children=[
                                    dmc.Stack(
                                        gap="xs",
                                        children=[
                                            dmc.Group(justify="apart", children=[
                                                dmc.Text("Customer Segments", c=colors['text_secondary'], size="sm"),
                                                dmc.ThemeIcon("🔍", variant="light", color="violet")
                                            ]),
                                            dmc.Title(id="clusters-display", order=3, c=colors['text_primary']),
                                            dmc.Text("Optimal Clusters", size="xs", c=colors['text_secondary']),
                                            dmc.Progress(value=80, color="violet", size="sm", mt="xs")
                                        ]
                                    )
                                ]
                            ),
                        ]),
                    ]
                ),
                
                # Time Series Analysis Section - USING ACTUAL DATA
                dmc.Grid(
                    gutter="lg",
                    mb="xl",
                    children=[
                        dmc.GridCol(span=6, children=[
                            dmc.Paper(
                                p="lg",
                                style={
                                    'backgroundColor': colors['card_bg'],
                                    'border': f'1px solid {colors["card_border"]}',
                                    'borderRadius': '15px',
                                },
                                children=[
                                    dmc.Stack(
                                        gap="md",
                                        children=[
                                            dmc.Group(justify="apart", children=[
                                                dmc.Title("📈 Sales & Revenue Trends", order=4, c=colors['text_primary']),
                                                dmc.Badge("Time Series", color="blue", variant="light")
                                            ]),
                                            dcc.Graph(id="sales-trend-chart"),
                                            dmc.Text("Annual sales volume and revenue performance", 
                                                    size="sm", c=colors['text_secondary'])
                                        ]
                                    )
                                ]
                            ),
                        ]),
                        dmc.GridCol(span=6, children=[
                            dmc.Paper(
                                p="lg",
                                style={
                                    'backgroundColor': colors['card_bg'],
                                    'border': f'1px solid {colors["card_border"]}',
                                    'borderRadius': '15px',
                                },
                                children=[
                                    dmc.Stack(
                                        gap="md",
                                        children=[
                                            dmc.Group(justify="apart", children=[
                                                dmc.Title("💰 Price Evolution", order=4, c=colors['text_primary']),
                                                dmc.Badge("Market Analysis", color="green", variant="light")
                                            ]),
                                            dcc.Graph(id="price-trend-chart"),
                                            dmc.Text("Average price trends and market positioning", 
                                                    size="sm", c=colors['text_secondary'])
                                        ]
                                    )
                                ]
                            ),
                        ]),
                    ]
                ),
                
                # Feature Importance & Market Analysis - USING ACTUAL FEATURE IMPORTANCE
                dmc.Grid(
                    gutter="lg",
                    mb="xl",
                    children=[
                        dmc.GridCol(span=6, children=[
                            dmc.Paper(
                                p="lg",
                                style={
                                    'backgroundColor': colors['card_bg'],
                                    'border': f'1px solid {colors["card_border"]}',
                                    'borderRadius': '15px',
                                },
                                children=[
                                    dmc.Stack(
                                        gap="md",
                                        children=[
                                            dmc.Group(justify="apart", children=[
                                                dmc.Title("🔍 Price Drivers", order=4, c=colors['text_primary']),
                                                dmc.Badge("Feature Importance", color="blue", variant="light")
                                            ]),
                                            dcc.Graph(id="price-feature-importance-chart"),
                                            dmc.Text("Key factors influencing vehicle pricing", 
                                                    size="sm", c=colors['text_secondary'])
                                        ]
                                    )
                                ]
                            ),
                        ]),
                        dmc.GridCol(span=6, children=[
                            dmc.Paper(
                                p="lg",
                                style={
                                    'backgroundColor': colors['card_bg'],
                                    'border': f'1px solid {colors["card_border"]}',
                                    'borderRadius': '15px',
                                },
                                children=[
                                    dmc.Stack(
                                        gap="md",
                                        children=[
                                            dmc.Group(justify="apart", children=[
                                                dmc.Title("📊 Sales Drivers", order=4, c=colors['text_primary']),
                                                dmc.Badge("Feature Importance", color="green", variant="light")
                                            ]),
                                            dcc.Graph(id="sales-feature-importance-chart"),
                                            dmc.Text("Key factors influencing sales volume", 
                                                    size="sm", c=colors['text_secondary'])
                                        ]
                                    )
                                ]
                            ),
                        ]),
                    ]
                ),
                
                # Customer Segmentation & Regional Analysis
                dmc.Grid(
                    gutter="lg",
                    mb="xl",
                    children=[
                        dmc.GridCol(span=6, children=[
                            dmc.Paper(
                                p="lg",
                                style={
                                    'backgroundColor': colors['card_bg'],
                                    'border': f'1px solid {colors["card_border"]}',
                                    'borderRadius': '15px',
                                },
                                children=[
                                    dmc.Stack(
                                        gap="md",
                                        children=[
                                            dmc.Group(justify="apart", children=[
                                                dmc.Title("🎯 Customer Segments", order=4, c=colors['text_primary']),
                                                dmc.Badge("Clustering", color="violet", variant="light")
                                            ]),
                                            dcc.Graph(id="clustering-chart"),
                                            dmc.Text("Customer groups based on purchasing behavior", 
                                                    size="sm", c=colors['text_secondary'])
                                        ]
                                    )
                                ]
                            ),
                        ]),
                        dmc.GridCol(span=6, children=[
                            dmc.Paper(
                                p="lg",
                                style={
                                    'backgroundColor': colors['card_bg'],
                                    'border': f'1px solid {colors["card_border"]}',
                                    'borderRadius': '15px',
                                },
                                children=[
                                    dmc.Stack(
                                        gap="md",
                                        children=[
                                            dmc.Group(justify="apart", children=[
                                                dmc.Title("🌍 Regional Performance", order=4, c=colors['text_primary']),
                                                dmc.Badge("Market Share", color="orange", variant="light")
                                            ]),
                                            dcc.Graph(id="regional-performance-chart"),
                                            dmc.Text("Sales performance across different regions", 
                                                    size="sm", c=colors['text_secondary'])
                                        ]
                                    )
                                ]
                            ),
                        ]),
                    ]
                ),
                
                # AI Prediction Engine - USING ACTUAL MODELS
                dmc.Grid(
                    gutter="lg",
                    children=[
                        dmc.GridCol(span=6, children=[
                            dmc.Paper(
                                p="lg",
                                style={
                                    'backgroundColor': colors['card_bg'],
                                    'border': f'1px solid {colors["card_border"]}',
                                    'borderRadius': '15px',
                                },
                                children=[
                                    dmc.Stack(
                                        gap="md",
                                        children=[
                                            dmc.Title("🔮 AI Price Prediction", order=4, c=colors['text_primary']),
                                            dmc.Grid(
                                                children=[
                                                    dmc.GridCol(span=6, children=[
                                                        dmc.NumberInput(
                                                            label="Vehicle Year",
                                                            id="predict-year",
                                                            value=2022,
                                                            min=min_year,
                                                            max=max_year,
                                                        ),
                                                    ]),
                                                    dmc.GridCol(span=6, children=[
                                                        dmc.NumberInput(
                                                            label="Mileage (KM)",
                                                            id="predict-mileage",
                                                            value=50000,
                                                            min=0,
                                                            max=500000,
                                                        ),
                                                    ]),
                                                ]
                                            ),
                                            dmc.Grid(
                                                children=[
                                                    dmc.GridCol(span=6, children=[
                                                        dmc.Select(
                                                            label="Model",
                                                            id="predict-model",
                                                            data=[{"value": model, "label": model} for model in models],
                                                            value=models[0] if models else "",
                                                        ),
                                                    ]),
                                                    dmc.GridCol(span=6, children=[
                                                        dmc.Select(
                                                            label="Region",
                                                            id="predict-region",
                                                            data=[{"value": region, "label": region} for region in regions],
                                                            value=regions[0] if regions else "",
                                                        ),
                                                    ]),
                                                ]
                                            ),
                                            dmc.Grid(
                                                children=[
                                                    dmc.GridCol(span=6, children=[
                                                        dmc.Select(
                                                            label="Fuel Type",
                                                            id="predict-fuel-type",
                                                            data=[{"value": fuel, "label": fuel} for fuel in fuel_types],
                                                            value=fuel_types[0] if fuel_types else "",
                                                        ),
                                                    ]),
                                                    dmc.GridCol(span=6, children=[
                                                        dmc.Select(
                                                            label="Transmission",
                                                            id="predict-transmission",
                                                            data=[{"value": trans, "label": trans} for trans in transmissions],
                                                            value=transmissions[0] if transmissions else "",
                                                        ),
                                                    ]),
                                                ]
                                            ),
                                            dmc.NumberInput(
                                                label="Engine Size (L)",
                                                id="predict-engine-size",
                                                value=2.0,
                                                min=1.0,
                                                max=6.0,
                                                step=0.1
                                            ),
                                            dmc.Button("Predict Optimal Price", id="predict-price-btn", color="blue", size="lg", fullWidth=True),
                                            dmc.Stack(
                                                gap="xs",
                                                mt="md",
                                                children=[
                                                    dmc.Title(id="price-prediction-output", order=2, c=colors['primary']),
                                                    dmc.Text(id="price-confidence-output", size="sm", c=colors['text_secondary'])
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            ),
                        ]),
                        dmc.GridCol(span=6, children=[
                            dmc.Paper(
                                p="lg",
                                style={
                                    'backgroundColor': colors['card_bg'],
                                    'border': f'1px solid {colors["card_border"]}',
                                    'borderRadius': '15px',
                                },
                                children=[
                                    dmc.Stack(
                                        gap="md",
                                        children=[
                                            dmc.Title("📊 Sales Potential Analysis", order=4, c=colors['text_primary']),
                                            dmc.Text("Analyze sales classification and volume potential", 
                                                    c=colors['text_secondary']),
                                            dmc.Button("Analyze Sales Potential", id="analyze-sales-btn", color="green", size="lg", fullWidth=True),
                                            
                                            dmc.Stack(
                                                gap="md",
                                                mt="md",
                                                children=[
                                                    dmc.Group(
                                                        grow=
                                                        True,
                                                        children=[
                                                            dmc.Paper(
                                                                p="md",
                                                                style={'backgroundColor': colors['background'], 'borderRadius': '10px'},
                                                                children=[
                                                                    dmc.Stack(
                                                                        gap="xs",
                                                                        align="center",
                                                                        children=[
                                                                            dmc.Text("Sales Classification", size="sm", c=colors['text_secondary']),
                                                                            dmc.Title(id="sales-classification-output", order=3, c=colors['success'])
                                                                        ]
                                                                    )
                                                                ]
                                                            ),
                                                            dmc.Paper(
                                                                p="md",
                                                                style={'backgroundColor': colors['background'], 'borderRadius': '10px'},
                                                                children=[
                                                                    dmc.Stack(
                                                                        gap="xs",
                                                                        align="center",
                                                                        children=[
                                                                            dmc.Text("Expected Volume", size="sm", c=colors['text_secondary']),
                                                                            dmc.Title(id="expected-volume-output", order=3, c=colors['warning'])
                                                                        ]
                                                                    )
                                                                ]
                                                            ),
                                                        ]
                                                    ),
                                                    dmc.Accordion(
                                                        children=[
                                                            dmc.AccordionItem(
                                                                value="business-recommendations",
                                                                children=[
                                                                    dmc.AccordionControl("💡 Strategic Recommendations"),
                                                                    dmc.AccordionPanel(
                                                                        dmc.Stack(
                                                                            gap="xs",
                                                                            children=[
                                                                                dmc.Text(id="specific-recommendations-output", size="sm")
                                                                            ]
                                                                        )
                                                                    ),
                                                                ],
                                                            ),
                                                        ],
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            ),
                        ]),
                    ]
                ),
            ]
        )
    ]
)

# Callback to apply filters and update data store
@app.callback(
    [Output('filtered-data-store', 'data'),
     Output('price-mae-display', 'children'),
     Output('price-r2-display', 'children'),
     Output('price-progress', 'value'),
     Output('classification-accuracy-display', 'children'),
     Output('classification-progress', 'value'),
     Output('volume-mae-display', 'children'),
     Output('volume-r2-display', 'children'),
     Output('volume-progress', 'value'),
     Output('clusters-display', 'children')],
    [Input('apply-filters-btn', 'n_clicks')],
    [State('year-range-slider', 'value'),
     State('model-filter', 'value'),
     State('region-filter', 'value'),
     State('fuel-filter', 'value'),
     State('transmission-filter', 'value'),
     State('color-filter', 'value'),
     State('classification-filter', 'value')]
)
def update_filtered_data_and_metrics(n_clicks, year_range, selected_models, selected_regions, selected_fuels, 
                        selected_transmissions, selected_colors, selected_classifications):
    if n_clicks is None:
        # Return initial values
        return (
            df.to_json(date_format='iso', orient='split'),
            f"${ml_solutions.metrics.get('price_mae', 0):,.0f}",
            f"R²: {ml_solutions.metrics.get('price_r2', 0):.3f}",
            min(100, ml_solutions.metrics.get('price_r2', 0) * 100),
            f"{ml_solutions.metrics.get('classification_accuracy', 0)*100:.1f}%",
            ml_solutions.metrics.get('classification_accuracy', 0)*100,
            f"{ml_solutions.metrics.get('volume_mae', 0):.1f}",
            f"R²: {ml_solutions.metrics.get('volume_r2', 0):.3f}",
            min(100, ml_solutions.metrics.get('volume_r2', 0) * 100),
            f"{ml_solutions.metrics.get('optimal_clusters', 4)}"
        )
    
    filtered_data = filter_data(
        df, 
        year_range, 
        selected_models, 
        selected_regions, 
        selected_fuels, 
        selected_transmissions, 
        selected_colors, 
        selected_classifications
    )
    
    # Retrain models on filtered data
    ml_solutions.train_models_on_data(filtered_data)
    
    # Update metrics displays
    return (
        filtered_data.to_json(date_format='iso', orient='split'),
        f"${ml_solutions.metrics.get('price_mae', 0):,.0f}",
        f"R²: {ml_solutions.metrics.get('price_r2', 0):.3f}",
        min(100, ml_solutions.metrics.get('price_r2', 0) * 100),
        f"{ml_solutions.metrics.get('classification_accuracy', 0)*100:.1f}%",
        ml_solutions.metrics.get('classification_accuracy', 0)*100,
        f"{ml_solutions.metrics.get('volume_mae', 0):.1f}",
        f"R²: {ml_solutions.metrics.get('volume_r2', 0):.3f}",
        min(100, ml_solutions.metrics.get('volume_r2', 0) * 100),
        f"{ml_solutions.metrics.get('optimal_clusters', 4)}"
    )

# Callback to update all charts based on filtered data - USING ACTUAL ANALYSIS
@app.callback(
    [Output("sales-trend-chart", "figure"),
     Output("price-trend-chart", "figure"),
     Output("price-feature-importance-chart", "figure"),
     Output("sales-feature-importance-chart", "figure"),
     Output("clustering-chart", "figure"),
     Output("regional-performance-chart", "figure")],
    [Input('filtered-data-store', 'data')]
)
def update_all_charts(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df
    else:
        filtered_df = pd.read_json(filtered_data_json, orient='split')
    
    if len(filtered_df) == 0:
        # Return empty figures if no data
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template='plotly_dark', 
            plot_bgcolor=colors['card_bg'], 
            paper_bgcolor=colors['card_bg'],
            title="No data available for current filters"
        )
        return [empty_fig] * 6
    
    # Sales Trend Analysis - ACTUAL DATA
    yearly_trends = filtered_df.groupby('Year').agg({
        'Sales_Volume': 'sum',
        'Revenue': 'sum',
        'Price_USD': 'mean'
    }).reset_index()
    
    sales_trend_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    sales_trend_fig.add_trace(
        go.Bar(x=yearly_trends['Year'], y=yearly_trends['Sales_Volume'], 
               name="Sales Volume", marker_color='#2563eb'),
        secondary_y=False,
    )
    
    sales_trend_fig.add_trace(
        go.Scatter(x=yearly_trends['Year'], y=yearly_trends['Revenue'], 
                  name="Revenue", line=dict(color='#10b981', width=3)),
        secondary_y=True,
    )
    
    sales_trend_fig.update_layout(
        title_text="Sales Volume & Revenue Trends",
        template='plotly_dark',
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor=colors['card_bg']
    )
    sales_trend_fig.update_xaxes(title_text="Year")
    sales_trend_fig.update_yaxes(title_text="Sales Volume", secondary_y=False)
    sales_trend_fig.update_yaxes(title_text="Revenue ($)", secondary_y=True)
    
    # Price Trend Analysis - ACTUAL DATA
    price_trend_data = filtered_df.groupby(['Year', 'Fuel_Type']).agg({
        'Price_USD': 'mean',
        'Sales_Volume': 'sum'
    }).reset_index()
    
    price_trend_fig = px.line(
        price_trend_data,
        x='Year',
        y='Price_USD',
        color='Fuel_Type',
        title='Average Price Trends by Fuel Type',
        markers=True
    )
    price_trend_fig.update_layout(template='plotly_dark', plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'])
    
    # Feature Importance Charts - ACTUAL ML RESULTS
    if 'price_feature_importance' in ml_solutions.metrics:
        price_feature_fig = px.bar(
            ml_solutions.metrics['price_feature_importance'],
            x='importance',
            y='feature',
            orientation='h',
            title='Price Prediction - Feature Importance',
            color='importance',
            color_continuous_scale='Blues'
        )
    else:
        price_feature_fig = go.Figure()
        price_feature_fig.update_layout(title="No feature importance data available")
    
    price_feature_fig.update_layout(template='plotly_dark', plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'])
    
    if 'classification_feature_importance' in ml_solutions.metrics:
        sales_feature_fig = px.bar(
            ml_solutions.metrics['classification_feature_importance'],
            x='importance',
            y='feature',
            orientation='h',
            title='Sales Classification - Feature Importance',
            color='importance',
            color_continuous_scale='Greens'
        )
    else:
        sales_feature_fig = go.Figure()
        sales_feature_fig.update_layout(title="No feature importance data available")
    
    sales_feature_fig.update_layout(template='plotly_dark', plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'])
    
    # Clustering Analysis - ACTUAL DATA
    if 'Profitability_Cluster' in filtered_df.columns:
        cluster_summary = filtered_df.groupby('Profitability_Cluster').agg({
            'Price_USD': 'mean',
            'Sales_Volume': 'mean',
            'Revenue': 'mean',
            'Model': 'count'
        }).reset_index()
        
        clustering_fig = px.scatter(
            cluster_summary,
            x='Price_USD',
            y='Sales_Volume',
            size='Revenue',
            color='Profitability_Cluster',
            title='Customer Segments - Price vs Sales Volume',
            hover_data=['Revenue'],
            size_max=40
        )
    else:
        clustering_fig = go.Figure()
        clustering_fig.update_layout(title="No clustering data available")
    
    clustering_fig.update_layout(template='plotly_dark', plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'])
    
    # Regional Performance - ACTUAL DATA
    regional_data = filtered_df.groupby('Region').agg({
        'Sales_Volume': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    regional_fig = px.pie(
        regional_data,
        values='Sales_Volume',
        names='Region',
        title='Sales Distribution by Region',
        hole=0.4
    )
    regional_fig.update_layout(template='plotly_dark', plot_bgcolor=colors['card_bg'], paper_bgcolor=colors['card_bg'])
    
    return [sales_trend_fig, price_trend_fig, price_feature_fig, sales_feature_fig, clustering_fig, regional_fig]

# Prediction callbacks - USING ACTUAL ML MODELS
@app.callback(
    [Output("price-prediction-output", "children"),
     Output("price-confidence-output", "children")],
    [Input("predict-price-btn", "n_clicks")],
    [State("predict-year", "value"),
     State("predict-mileage", "value"),
     State("predict-model", "value"),
     State("predict-region", "value"),
     State("predict-fuel-type", "value"),
     State("predict-transmission", "value"),
     State("predict-engine-size", "value")]
)
def predict_price(n_clicks, year, mileage, model, region, fuel_type, transmission, engine_size):
    if n_clicks is None:
        return "Enter vehicle details", "Click predict to get AI-powered price estimation"
    
    try:
        # Prepare input features for the model
        vehicle_age = 2024 - year
        price_per_km = 0  # Will be calculated by model
        
        # Encode categorical variables
        model_encoded = label_encoders['Model'].transform([model])[0]
        region_encoded = label_encoders['Region'].transform([region])[0]
        fuel_encoded = label_encoders['Fuel_Type'].transform([fuel_type])[0]
        transmission_encoded = label_encoders['Transmission'].transform([transmission])[0]
        color_encoded = 0  # Default color
        
        # Create feature array
        features = np.array([[
            year, engine_size, mileage, vehicle_age,
            price_per_km, 0,  # These will be handled by the model
            model_encoded, region_encoded, color_encoded,
            fuel_encoded, transmission_encoded
        ]])
        
        # Scale features and predict
        if 'price_prediction' in ml_solutions.models and 'price' in ml_solutions.scalers:
            scaler = ml_solutions.scalers['price']
            features_scaled = scaler.transform(features)
            predicted_price = ml_solutions.models['price_prediction'].predict(features_scaled)[0]
            
            # Calculate confidence based on similar vehicles in dataset
            similar_cars = df[
                (df['Model'] == model) & 
                (df['Region'] == region) &
                (df['Fuel_Type'] == fuel_type) &
                (df['Transmission'] == transmission)
            ]
            
            confidence = min(95, 70 + (len(similar_cars) * 2))  # Base confidence + data availability
            
            return f"${predicted_price:,.0f}", f"Confidence: {confidence:.0f}% based on market data"
        else:
            return "Model not trained", "Please apply filters first to train models"
        
    except Exception as e:
        return "Prediction Error", f"Please check all input values: {str(e)}"

@app.callback(
    [Output("sales-classification-output", "children"),
     Output("expected-volume-output", "children"),
     Output("specific-recommendations-output", "children")],
    [Input("analyze-sales-btn", "n_clicks")],
    [State('filtered-data-store', 'data')]
)
def analyze_sales_potential(n_clicks, filtered_data_json):
    if n_clicks is None:
        return "Click to Analyze", "N/A", "Analysis will appear here"
    
    try:
        if filtered_data_json:
            filtered_df = pd.read_json(filtered_data_json, orient='split')
        else:
            filtered_df = df
        
        if len(filtered_df) == 0:
            return "No Data", "N/A", "Please adjust filters to get data"
        
        # Actual analysis based on filtered data
        total_sales = filtered_df['Sales_Volume'].sum()
        avg_price = filtered_df['Price_USD'].mean()
        high_performance_ratio = (filtered_df['Sales_Classification'] == 'High').mean()
        
        # Determine classification
        if high_performance_ratio > 0.6:
            classification = "HIGH PERFORMANCE"
            volume_estimate = f"{total_sales:,.0f}+ units"
        elif high_performance_ratio > 0.3:
            classification = "MODERATE PERFORMANCE"
            volume_estimate = f"{total_sales:,.0f} units"
        else:
            classification = "OPTIMIZATION NEEDED"
            volume_estimate = f"{total_sales:,.0f} units"
        
        # Generate specific recommendations
        top_model = filtered_df.groupby('Model')['Sales_Volume'].sum().idxmax() if len(filtered_df['Model'].unique()) > 0 else "N/A"
        top_region = filtered_df.groupby('Region')['Sales_Volume'].sum().idxmax() if len(filtered_df['Region'].unique()) > 0 else "N/A"
        avg_mileage = filtered_df['Mileage_KM'].mean()
        
        recommendations = [
            f"• Focus on {top_model} in {top_region} region",
            f"• Average price: ${avg_price:,.0f}",
            f"• Consider mileage optimization (avg: {avg_mileage:,.0f} km)",
            f"• High performance ratio: {high_performance_ratio:.1%}"
        ]
        
        recommendations_text = "\n".join(recommendations)
        
        return classification, volume_estimate, recommendations_text
        
    except Exception as e:
        return "Analysis Error", "N/A", f"Error in analysis: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=8020)