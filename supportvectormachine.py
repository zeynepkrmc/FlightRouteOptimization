import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df['DistanceKilometers'] = df['DistanceKilometers'].str.replace(',', '').astype(float)
    df['FlightDelayMin'] = df['FlightDelayMin'].astype(float)
    df['FlightDelayMin'] = df['FlightDelayMin'].fillna(0)
    df['DelayStatus'] = df['FlightDelayMin'].apply(lambda x: 1 if x > 0 else 0)
    df['DelayCategory'] = pd.cut(df['FlightDelayMin'], bins=[-1, 0, 15, 60, float('inf')], 
                                 labels=['No Delay', 'Short Delay', 'Medium Delay', 'Long Delay'])

    if df['DistanceKilometers'].dtype == 'object':
        df['DistanceKilometers'] = df['DistanceKilometers'].str.replace(',', '').astype(float)

    df['FlightDelayMin'] = df['FlightDelayMin'].astype(float)
    df['FlightDelayMin'] = df['FlightDelayMin'].fillna(0)

    return df
def train_svm(df):
    features = ['DistanceKilometers']
    target = 'DelayStatus'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    model = SVC(kernel='linear', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")
    
    model.fit(X_train_balanced, y_train_balanced)

    y_train_pred = model.predict(X_train_balanced)
    train_accuracy = accuracy_score(y_train_balanced, y_train_pred)
    print(f"Train Accuracy: {train_accuracy:.4f}")

    y_pred = model.predict(X_test_scaled)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler, X_test, y_test, y_pred
def create_graph_with_delay(df):
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        if row['OriginAirportID'] != row['DestAirportID']:
            distance = row['DistanceKilometers']
            delay = row['FlightDelayMin']
            weight = distance + delay
            
            G.add_edge(row['OriginAirportID'], row['DestAirportID'], weight=weight, distance=distance, delay=delay)
    
    return G
def find_and_display_fastest_path(graph, origin, destination):
    try:
        fastest_path = nx.dijkstra_path(graph, origin, destination, weight='weight')
        total_distance = sum(graph[u][v]['distance'] for u, v in zip(fastest_path[:-1], fastest_path[1:]))
        total_delay = sum(graph[u][v]['delay'] for u, v in zip(fastest_path[:-1], fastest_path[1:]))
        
        print(f"{origin} ve {destination} arasındaki en hızlı yol: {fastest_path}")
        print(f"Toplam Mesafe: {total_distance:.2f} km")
        print(f"Toplam Tahmini Gecikme: {total_delay:.2f} dakika")
        
        return fastest_path, total_distance, total_delay
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

def train_and_visualize_svr(df):
    features = ['DistanceKilometers']
    target = 'FlightDelayMin'

    X = df[features].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    epsilon_range = np.linspace(0.1, 1.0, 10)  
    C_range = np.logspace(-1, 2, 10)         
    mse_matrix = np.zeros((len(C_range), len(epsilon_range)))

    for i, C in enumerate(C_range):
        for j, epsilon in enumerate(epsilon_range):
            svr = SVR(kernel='rbf', C=C, epsilon=epsilon)
            svr.fit(X_train, y_train)
            y_pred = svr.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mse_matrix[i, j] = mse
            overall_mse = mse_matrix.mean()
    print(f"Genel MSE (Overall MSE): {overall_mse:.4f}")
    C_mesh, epsilon_mesh = np.meshgrid(C_range, epsilon_range, indexing='ij')

    plt.figure(figsize=(10, 6))
    contour = plt.contourf(C_mesh, epsilon_mesh, mse_matrix, levels=20, cmap='Blues')
    plt.colorbar(contour)
    plt.xscale('log')
    plt.title("SVR Performans Optimizasyonu (C ve Epsilon)")
    plt.xlabel("C (Regularization Parameter)")
    plt.ylabel("Epsilon (Epsilon-tube)")
    plt.show()

if __name__ == "__main__":
    file_path = 'data/flights.csv'
    data = load_and_clean_data(file_path)
    
    model, scaler, X_test, y_test, y_pred = train_svm(data)
    
    train_and_visualize_svr(data)

    origin = input("Başlangıç havalimanı kodunu giriniz: ")
    destination = input("Varış havalimanı kodunu giriniz: ")
    
    graph = create_graph_with_delay(data)
    
    fastest_path, total_distance, total_delay = find_and_display_fastest_path(graph, origin, destination)
    
    if fastest_path:
        plot_graph_with_details(graph, path=fastest_path, origin=origin, destination=destination)
