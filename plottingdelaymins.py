import pandas as pd
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/mdrilwan/datasets/master/flights.csv'
data = pd.read_csv(url)

data = data.dropna(subset=['DistanceKilometers', 'FlightDelayMin'])  
data['DistanceKilometers'] = pd.to_numeric(data['DistanceKilometers'], errors='coerce') 
data = data.dropna(subset=['DistanceKilometers', 'FlightDelayMin']) 

plt.figure(figsize=(10, 6))
plt.scatter(data['DistanceKilometers'], data['FlightDelayMin'], 
            color='blue', alpha=1, s=100,  
            edgecolors='blue', linewidth=1.5, 
            marker='o')  
plt.title("Flight Delay by Distance (Kilometers)", fontsize=16)
plt.xlabel("Distance (Kilometers)", fontsize=14)
plt.ylabel("Flight Delay (Minutes)", fontsize=14)
plt.grid(False) 
plt.tight_layout()
plt.show()
