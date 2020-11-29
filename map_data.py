import pandas as pd

fires = pd.read_csv('fires_point.csv', usecols=['X', 'Y'])
fires = fires.rename(columns={'X': 'longitude', 'Y': 'latitude'})

#fires = fires[:5000]

aggregated_fires = pd.DataFrame(columns=['longitude', 'latitude'])
print(aggregated_fires)