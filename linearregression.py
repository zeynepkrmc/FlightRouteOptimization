import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
 
    df['DistanceKilometers'] = df['DistanceKilometers'].str.replace(',', '').astype(float)
    df['FlightDelayMin'] = df['FlightDelayMin'].astype(float)
    df['FlightDelayMin'] = df['FlightDelayMin'].fillna(0)

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

    y_train_pred = (model.predict(X_train_scaled) > 0.5).astype(int)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Train Accuracy: {train_accuracy:.4f}")

    y_test_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_cv_pred = cross_val_predict(model, X, y, cv=skf, method='predict')
    y_cv_pred = (y_cv_pred > 0.5).astype(int)
    cv_accuracy = accuracy_score(y, y_cv_pred)
    print(f"Cross-Validation Accuracy: {cv_accuracy:.4f}")

    overall_mse = mean_squared_error(y, model.predict(scaler.transform(X)))
    print(f"Overall MSE: {overall_mse:.4f}")

    return model, scaler

def create_graph_with_delay(df):
    G = nx.DiGraph()

    for _, row in df.iterrows():
        if row['OriginAirportID'] != row['DestAirportID']:  
            distance = row['DistanceKilometers']
            delay = row['FlightDelayMin']
            weight = distance + delay

            G.add_edge(row['OriginAirportID'], row['DestAirportID'], weight=weight, distance=distance, delay=delay)

    return G

def find_and_display_shortest_path(graph, origin, destination):
    try:
        shortest_path = nx.dijkstra_path(graph, origin, destination, weight='weight')
        total_distance = sum(graph[u][v]['distance'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
        total_delay = sum(graph[u][v]['delay'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
        return shortest_path, total_distance, total_delay
    except nx.NetworkXNoPath:
        print(f"{origin} ile {destination} arasında bir yol bulunmamaktadır.")
        return None, None, None

def plot_graph_with_details(graph, path=None, origin=None, destination=None):
    if path:
        subgraph = graph.subgraph(path)
    else:
        subgraph = graph

    pos = nx.spring_layout(subgraph, seed=42)
    plt.figure(figsize=(10, 10))

    if path:
        edges_in_path = list(zip(path, path[1:]))
        nx.draw_networkx_edges(subgraph, pos, edgelist=edges_in_path, edge_color='red', width=2)

    nx.draw_networkx_edges(subgraph, pos, edge_color='gray', alpha=0.5)
    node_labels = {node: node for node in subgraph.nodes}
    nx.draw_networkx_nodes(subgraph, pos, node_size=800, node_color='skyblue')
    nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=12, font_weight='bold')

    if origin and destination:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=[origin, destination], node_color='orange', node_size=800)

    plt.title("Flight Route Graph with Delay and Distance")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    file_path = 'data/flights.csv' 

    data = load_and_clean_data(file_path)

    model, scaler = train_linear_regression(data)

    origin = input("Başlangıç havalimanı kodunu giriniz: ")
    destination = input("Varış havalimanı kodunu giriniz: ")

    graph = create_graph_with_delay(data)

    shortest_path, total_distance, total_delay = find_and_display_shortest_path(graph, origin, destination)

    if shortest_path:
        print(f"{origin} ve {destination} arasındaki en kısa yol: {shortest_path}")
        print(f"Toplam Mesafe: {total_distance:.2f} km")
        print(f"Toplam Tahmini Gecikme: {total_delay:.2f} dakika")


        plot_graph_with_details(graph, path=shortest_path, origin=origin, destination=destination)
        delay_prediction = (model.predict(scaler.transform(pd.DataFrame([[total_distance]], columns=['DistanceKilometers']))) > 0.5).astype(int)
        delay_status = "Gecikme" if delay_prediction[0] == 1 else "Gecikme Yok"
        print(f"Tahmini durum: {delay_status}")
