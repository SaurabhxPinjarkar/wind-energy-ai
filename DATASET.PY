import pandas as pd
import requests
import time

# Load your file
df_locations = pd.read_csv("S:\AI_PROJECT\maharashtra_100_locations_named.csv")

all_data = []

for index, row in df_locations.iterrows():
    lat = row["Latitude"]
    lon = row["Longitude"]
    place = row["Place"]
    
    print(f"Fetching: {place}")
    
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=WS10M,T2M,PS,RH2M&community=RE&longitude={lon}&latitude={lat}&start=20200101&end=20231231&format=JSON"
    
    response = requests.get(url).json()
    
    try:
        data = response['properties']['parameter']
        
        temp_df = pd.DataFrame({
            "Date": data['WS10M'].keys(),
            "WS10M": data['WS10M'].values(),
            "T2M": data['T2M'].values(),
            "PS": data['PS'].values(),
            "RH2M": data['RH2M'].values()
        })
        
        temp_df["Place"] = place
        temp_df["Latitude"] = lat
        temp_df["Longitude"] = lon
        
        all_data.append(temp_df)
        
    except:
        print(f"Error at {place}")
    
    time.sleep(0.5)  # avoid rate limit

# Combine all
final_df = pd.concat(all_data)

# Save final dataset
final_df.to_csv("final_wind_dataset.csv", index=False)

print("DONE ✅ Download from Files")