import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

print("="*60)
print("🚀 TRAVEL AI MODEL TRAINING")
print("="*60)

# Load dataset
print("\n📂 Loading dataset...")
df = pd.read_csv('dataset/travel_data.csv')
print(f"✅ Loaded {len(df)} destinations with {len(df.columns)} features")

# Feature Engineering
print("\n🔧 Encoding categorical features...")
le_type = LabelEncoder()
le_cost = LabelEncoder()
le_season = LabelEncoder()

df['type_encoded'] = le_type.fit_transform(df['type'])
df['cost_encoded'] = le_cost.fit_transform(df['cost_level'])
df['season_encoded'] = le_season.fit_transform(df['best_season'])

# Select features for recommendation
feature_columns = [
    'avg_temp', 'safety_rating', 'tourist_rating', 'nightlife_score',
    'culture_score', 'nature_score', 'adventure_score', 'family_friendly',
    'food_rating', 'type_encoded', 'cost_encoded', 'season_encoded'
]

X = df[feature_columns]
y = df['popularity_score']

print(f"✅ Using {len(feature_columns)} features for training")

# Normalize features
print("\n⚙️ Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train K-Nearest Neighbors for recommendation
print("\n🎯 Training K-NN Recommendation Model...")
knn_model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
knn_model.fit(X_scaled)
print("✅ K-NN model trained")

# Train Random Forest for popularity prediction
print("\n🌲 Training Random Forest Regressor...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Random Forest trained")
print(f"\n📊 Model Performance:")
print(f"   RMSE: {rmse:.2f}")
print(f"   R² Score: {r2:.3f}")

# Save models
print("\n💾 Saving models...")
os.makedirs('models', exist_ok=True)

model_artifacts = {
    'knn_model': knn_model,
    'rf_model': rf_model,
    'scaler': scaler,
    'label_encoders': {
        'type': le_type,
        'cost': le_cost,
        'season': le_season
    },
    'feature_columns': feature_columns,
    'dataset': df,
    'performance': {
        'rmse': rmse,
        'r2_score': r2
    }
}

with open('models/recommendation_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

print("✅ Model saved: models/recommendation_model.pkl")
print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print("\n✨ Ready to run: streamlit run app.py")
