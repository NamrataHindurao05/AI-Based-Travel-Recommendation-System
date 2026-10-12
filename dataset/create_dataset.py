import pandas as pd
import numpy as np

# Create comprehensive travel dataset
np.random.seed(42)

print("🚀 Creating Travel Dataset...")

cities = [
    "Paris", "Tokyo", "New York", "Barcelona", "Dubai", "London", "Rome", 
    "Singapore", "Istanbul", "Bangkok", "Sydney", "Amsterdam", "Prague",
    "Vienna", "Copenhagen", "Lisbon", "Budapest", "Krakow", "Edinburgh",
    "Dublin", "Reykjavik", "Stockholm", "Oslo", "Helsinki", "Berlin",
    "Munich", "Zurich", "Geneva", "Venice", "Florence", "Milan",
    "Madrid", "Seville", "Porto", "Athens", "Santorini", "Mykonos",
    "Marrakech", "Cairo", "Cape Town", "Nairobi", "Mumbai", "Delhi",
    "Jaipur", "Agra", "Goa", "Bali", "Phuket", "Hanoi", "Ho Chi Minh",
    "Seoul", "Busan", "Kyoto", "Osaka", "Hong Kong", "Macau", "Shanghai",
    "Beijing", "Xi'an", "Kuala Lumpur", "Penang", "Melbourne", "Auckland",
    "Queenstown", "Fiji", "Maldives", "Seychelles", "Mauritius", "Bora Bora",
    "Santorini", "Dubrovnik"
]

countries = [
    "France", "Japan", "USA", "Spain", "UAE", "UK", "Italy",
    "Singapore", "Turkey", "Thailand", "Australia", "Netherlands", "Czech Republic",
    "Austria", "Denmark", "Portugal", "Hungary", "Poland", "Scotland",
    "Ireland", "Iceland", "Sweden", "Norway", "Finland", "Germany",
    "Germany", "Switzerland", "Switzerland", "Italy", "Italy", "Italy",
    "Spain", "Spain", "Portugal", "Greece", "Greece", "Greece",
    "Morocco", "Egypt", "South Africa", "Kenya", "India", "India",
    "India", "India", "India", "Indonesia", "Thailand", "Vietnam", "Vietnam",
    "South Korea", "South Korea", "Japan", "Japan", "Hong Kong", "Macau", "China",
    "China", "China", "Malaysia", "Malaysia", "Australia", "New Zealand",
    "New Zealand", "Fiji", "Maldives", "Seychelles", "Mauritius", "French Polynesia",
    "Greece", "Croatia"
]

types = [
    "Cultural", "Cultural", "Urban", "Beach", "Urban", "Historical", "Historical",
    "Urban", "Cultural", "Beach", "Beach", "Cultural", "Historical",
    "Cultural", "Urban", "Beach", "Historical", "Historical", "Historical",
    "Cultural", "Nature", "Urban", "Nature", "Urban", "Urban",
    "Cultural", "Mountain", "Urban", "Historical", "Cultural", "Urban",
    "Cultural", "Cultural", "Beach", "Historical", "Beach", "Beach",
    "Cultural", "Historical", "Adventure", "Adventure", "Cultural", "Urban",
    "Cultural", "Historical", "Beach", "Beach", "Beach", "Cultural", "Urban",
    "Urban", "Beach", "Cultural", "Urban", "Urban", "Urban", "Urban",
    "Historical", "Historical", "Urban", "Beach", "Urban", "Nature",
    "Adventure", "Beach", "Beach", "Beach", "Beach", "Beach",
    "Beach", "Beach"
]

# Generate realistic features
n = len(cities)

data = {
    'city': cities,
    'country': countries,
    'type': types,
    'avg_temp': np.random.randint(15, 35, n),
    'cost_level': np.random.choice(['Budget', 'Mid-range', 'Luxury'], n, p=[0.3, 0.5, 0.2]),
    'safety_rating': np.round(np.random.uniform(3.5, 5.0, n), 1),
    'tourist_rating': np.round(np.random.uniform(3.5, 5.0, n), 1),
    'nightlife_score': np.random.randint(1, 11, n),
    'culture_score': np.random.randint(1, 11, n),
    'nature_score': np.random.randint(1, 11, n),
    'adventure_score': np.random.randint(1, 11, n),
    'family_friendly': np.random.randint(1, 11, n),
    'food_rating': np.round(np.random.uniform(3.5, 5.0, n), 1),
    'best_season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter', 'Year-round'], n),
    'avg_stay_days': np.random.randint(2, 8, n),
    'popularity_score': np.random.randint(50, 100, n)
}

# Adjust specific cities for realism
adjustments = {
    'Paris': {'culture_score': 10, 'tourist_rating': 4.8, 'cost_level': 'Luxury'},
    'Tokyo': {'culture_score': 10, 'safety_rating': 5.0, 'food_rating': 4.9},
    'Dubai': {'nightlife_score': 9, 'cost_level': 'Luxury', 'safety_rating': 4.8},
    'Bali': {'beach_score': 10, 'nature_score': 10, 'cost_level': 'Budget'},
}

df = pd.DataFrame(data)

# Save dataset
df.to_csv('travel_data.csv', index=False)

print(f"✅ Dataset Created Successfully!")
print(f"\n📊 Dataset Stats:")
print(f"   Total Destinations: {len(df)}")
print(f"   Features: {len(df.columns)}")
print(f"   Types: {df['type'].unique()}")
print(f"\n💾 Saved to: dataset/travel_data.csv")
print(f"\n🔍 Sample Data:")
print(df.head())
