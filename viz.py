import networkx as nx

def node_label(node):
    label = ""
    label += f"shape: {node.shape}"
    if len(node.inputs) > 0:
        label += f"\nop: {node.op.__class__.__name__}"
    return label

def graph(tensor):
    G = nx.DiGraph()
    visited = set()
    queue = [tensor]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            G.add_node(node, label=node_label(node))
            if node == tensor:
                G.nodes[node]['color'] = 'red' # add color to root node
            elif len(node.inputs) == 0:
                G.nodes[node]['color'] = 'green' # add color to leaf node
            else:
                G.nodes[node]['color'] = 'blue'
            for input_node in node.inputs:
                G.add_edge(input_node, node, label=f"{input_node.shape}")
                queue.append(input_node)

def draw(G):
    nx.draw(G, 
            node_color=[G.nodes[node].get('color') for node in G.nodes()],
            pos=nx.spectral_layout(G),
            labels=nx.get_node_attributes(G, 'label')
            )

def viz(tensor):
    draw(graph(tensor))