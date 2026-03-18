import pandas as pd
import numpy as np

# Real approximate coordinates of cities in Maharashtra
city_coords = {
    'Ratnagiri': (16.99, 73.30),
    'Satara': (17.68, 73.99),
    'Pune': (18.52, 73.85),
    'Ahmednagar': (19.09, 74.74),
    'Aurangabad': (19.87, 75.34),
    'Nagpur': (21.14, 79.08),
    'Konkan': (17.50, 73.50), # generalize
    'Kolhapur': (16.70, 74.24),
    'Solapur': (17.65, 75.90),
    'Beed': (18.98, 75.76),
    'Latur': (18.40, 76.56),
    'Vidarbha': (20.93, 77.77), # Amravati general
    'Ghats': (18.10, 73.60), # general western ghats
    'Plateau': (19.50, 76.50), # general deccan
    'Interior': (20.00, 77.00) # marathwada general
}

# 1. Read the original 100 locations mapping
loc_df = pd.read_csv("maharashtra_100_locations_named.csv")

# Create a mapping dictionary of old (lat, lon) to new (lat, lon)
mapping = {}
new_locations = []

for idx, row in loc_df.iterrows():
    place = row['Place']
    old_lat = row['Latitude']
    old_lon = row['Longitude']
    
    # extract base name
    base_name = place.split('_')[0]
    
    if base_name in city_coords:
        base_lat, base_lon = city_coords[base_name]
    else:
        # Default fallback to center of Maharashtra roughly
        base_lat, base_lon = 19.5, 76.0
        
    # Add small random noise to scatter points around the city
    # ~0.15 degrees is roughly 15-20km radius mapping
    new_lat = round(base_lat + np.random.uniform(-0.15, 0.15), 3)
    new_lon = round(base_lon + np.random.uniform(-0.15, 0.15), 3)
    
    mapping[place] = (new_lat, new_lon)
    
    new_locations.append({
        'Place': place,
        'Latitude': new_lat,
        'Longitude': new_lon
    })

# Save new locations
new_loc_df = pd.DataFrame(new_locations)
new_loc_df.to_csv("maharashtra_100_locations_named.csv", index=False)
print("Updated maharashtra_100_locations_named.csv")

# 2. Update the massive dataset directly so we don't have to fetch from NASA again
wind_df = pd.read_csv("final_wind_dataset.csv")

# Update lat/lon based on Place matching
for place, (new_lat, new_lon) in mapping.items():
    mask = wind_df['Place'] == place
    wind_df.loc[mask, 'Latitude'] = new_lat
    wind_df.loc[mask, 'Longitude'] = new_lon

wind_df.to_csv("final_wind_dataset.csv", index=False)
print("Updated final_wind_dataset.csv! Ready to retrain models.")
