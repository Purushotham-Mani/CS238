import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(graph, var_names):
    """
    Visualize the given DAG using NetworkX and Matplotlib.
    
    Parameters:
    -----------
    graph : Graph
        An instance of the Graph class containing the adjacency list.
    """
    # Create a directed graph object in NetworkX
    G_nx = nx.DiGraph()

    # Add nodes and edges to the graph
    for node in graph.adj_list:
        G_nx.add_node(var_names[node])  # Add each node

        for neighbor in graph.adj_list[node]:
            G_nx.add_edge(var_names[node], var_names[neighbor])  # Add edges from the adjacency list

    # Draw the graph using NetworkX's built-in drawing function
    plt.figure(figsize=(10, 6))
    pos = nx.circular_layout(G_nx)  # Compute layout for the graph
    nx.draw(G_nx, pos, with_labels=True, node_color='skyblue', 
            edge_color='gray', node_size=3000, font_size=15, 
            font_weight='bold', arrowsize=20)

    plt.title("Bayesian Network Graph", fontsize=20)
    plt.show()

