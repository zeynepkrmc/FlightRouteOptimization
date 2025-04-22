import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*StandardScaler.*")

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df['DistanceKilometers'] = df['DistanceKilometers'].str.replace(',', '').astype(float)
    df['AvgTicketPrice'] = df['AvgTicketPrice'].replace(r'[\$,]', '', regex=True).astype(float)
    df['FlightDelayMin'] = df['FlightDelayMin'].astype(float).fillna(0)
    df['FlightDelay'] = df['FlightDelay'].apply(lambda x: 1 if x == 'TRUE' else 0)
    df['DestWeather'] = df['DestWeather'].fillna('Unknown')
    df['OriginWeather'] = df['OriginWeather'].fillna('Unknown')
    return df

def train_linear_regression(df):
    features = ['DistanceKilometers']
    target = 'FlightDelayMin'
    X = df[features]
    y = (df[target] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler

def train_decision_tree(df):
    features = ['DestWeather', 'OriginWeather', 'FlightDelay', 'FlightDelayMin']
    target = 'FlightDelayType'
    label_encoders = {}
    for col in ['DestWeather', 'OriginWeather', target]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)
    return model, label_encoders[target]

def create_graph_with_delay(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        if row['OriginAirportID'] != row['DestAirportID']:
            weight = row['DistanceKilometers'] + row['FlightDelayMin']
            G.add_edge(row['OriginAirportID'], row['DestAirportID'],
                       weight=weight, distance=row['DistanceKilometers'], delay=row['FlightDelayMin'])
    return G

def find_and_display_fastest_path(graph, origin, destination, max_alternatives=3):
    try:

        fastest_path = nx.dijkstra_path(graph, origin, destination, weight='weight')
        total_distance = sum(graph[u][v]['distance'] for u, v in zip(fastest_path[:-1], fastest_path[1:]))
        total_delay = sum(graph[u][v]['delay'] for u, v in zip(fastest_path[:-1], fastest_path[1:]))

        alternative_paths = []
        all_paths = list(nx.all_simple_paths(graph, origin, destination))
        all_paths.sort(key=lambda path: sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:])))
        for path in all_paths:
            if path != fastest_path and len(alternative_paths) < max_alternatives:
                alternative_paths.append(path)

        return fastest_path, total_distance, total_delay, alternative_paths

    except nx.NetworkXNoPath:
        print(f"No path found between {origin} and {destination}.")
        return None, None, None, None

def predict_and_plot(graph, path, total_distance, linear_model, scaler, alternative_paths=None):
    print(f"Optimal Route: {path}")
    print(f"Total Distance for Optimal Route: {total_distance:.2f} km")
    
    X_scaled = scaler.transform([[total_distance]])
    delay_prediction = linear_model.predict(X_scaled)
    delay_status = "Delay" if delay_prediction[0] >= 1 else "No Delay"
    print(f"Predicted Delay (min) for Optimal Route: {delay_prediction[0]:.2f}")
    print(f"Delay Status for Optimal Route: {delay_status}")

    if alternative_paths:
        print("\nAlternative Routes:")
        for i, alt_path in enumerate(alternative_paths):
            alt_distance = sum(graph[u][v]['distance'] for u, v in zip(alt_path[:-1], alt_path[1:]))
            alt_delay = sum(graph[u][v]['delay'] for u, v in zip(alt_path[:-1], alt_path[1:]))
            print(f"Alternative Route {i + 1}: {alt_path}")
            print(f"  Total Distance: {alt_distance:.2f} km")
            print(f"  Total Delay: {alt_delay:.2f} minutes\n")
    
    plt.figure(figsize=(14, 10))
    
    subgraph_nodes = set(path)
    if alternative_paths:
        for alt_path in alternative_paths:
            subgraph_nodes.update(alt_path)
    subgraph = graph.subgraph(subgraph_nodes)
    
    pos = nx.kamada_kawai_layout(subgraph) 
 
    edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(subgraph, pos, edgelist=edges, edge_color='red', width=4, label="Optimal Path", alpha=0.7, arrows=True)

    if alternative_paths:
        for alt_path in alternative_paths:
            alt_edges = list(zip(alt_path[:-1], alt_path[1:]))
            nx.draw_networkx_edges(subgraph, pos, edgelist=alt_edges, edge_color='green', width=3, alpha=0.7, label="Alternative Path", arrows=True)

    nx.draw(subgraph, pos, node_size=1000, node_color='skyblue', with_labels=True, font_size=12, font_weight="bold")

    plt.title(f"Flight Paths from {path[0]} to {path[-1]}", fontsize=16, fontweight='bold')
    plt.legend(loc="upper left", fontsize=12)
    plt.axis('off')  
    plt.show()

if __name__ == "__main__":
    file_path = 'data/flights.csv'
    data = load_and_clean_data(file_path)

    linear_model, scaler = train_linear_regression(data)

    graph = create_graph_with_delay(data)

    origin = input("Origin Airport ID: ")
    destination = input("Destination Airport ID: ")
    fastest_path, total_distance, total_delay, alternative_paths = find_and_display_fastest_path(graph, origin, destination)

    if fastest_path:
        predict_and_plot(
            graph,
            path=fastest_path,
            total_distance=total_distance,
            linear_model=linear_model,
            scaler=scaler,
            alternative_paths=alternative_paths
        )
