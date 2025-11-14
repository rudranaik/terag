"""
TERAG Graph Visualization Utilities

Provides multiple visualization options for TERAG graphs:
1. Matplotlib static plots
2. Plotly interactive plots  
3. Pyvis network graphs
4. Export to various formats (GraphML, GML, etc.)
"""

import json
import warnings
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mpatches = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot as plotly_plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    plotly_plot = None

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    Network = None

from graph_builder import TERAGGraph


class GraphVisualizer:
    """
    Visualize TERAG graphs using multiple backends
    """
    
    def __init__(self, terag_graph: TERAGGraph):
        """
        Initialize visualizer with a TERAG graph
        
        Args:
            terag_graph: TERAGGraph instance
        """
        self.terag_graph = terag_graph
        self.networkx_graph = None
        
    def _ensure_networkx_graph(self, include_edge_weights: bool = False):
        """Ensure NetworkX graph is available"""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for visualization. Install with: pip install networkx")
        
        if self.networkx_graph is None:
            self.networkx_graph = self.terag_graph.to_networkx(
                include_node_attributes=True,
                include_edge_weights=include_edge_weights
            )
    
    def matplotlib_plot(
        self,
        figsize: Tuple[int, int] = (12, 8),
        node_size: int = 300,
        font_size: int = 8,
        save_path: Optional[str] = None,
        layout: str = "spring",
        show_labels: bool = True
    ) -> None:
        """
        Create static matplotlib visualization
        
        Args:
            figsize: Figure size
            node_size: Node size
            font_size: Font size for labels
            save_path: Path to save plot
            layout: Layout algorithm ('spring', 'kamada_kawai', 'circular', 'spectral')
            show_labels: Whether to show node labels
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required. Install with: pip install matplotlib")
            
        self._ensure_networkx_graph()
        G = self.networkx_graph
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Separate node types
        passage_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'passage']
        concept_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'concept']
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=passage_nodes, 
                              node_color='lightblue', node_size=node_size, 
                              label='Passages', alpha=0.7)
        nx.draw_networkx_nodes(G, pos, nodelist=concept_nodes, 
                              node_color='lightcoral', node_size=node_size//2, 
                              label='Concepts', alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
        
        # Draw labels if requested
        if show_labels:
            # Create labels (shortened for readability)
            labels = {}
            for node, data in G.nodes(data=True):
                if data.get('node_type') == 'passage':
                    labels[node] = f"P{data.get('chunk_index', '')}"
                else:
                    concept_text = data.get('concept_text', str(node))
                    labels[node] = concept_text[:10] + "..." if len(concept_text) > 10 else concept_text
            
            nx.draw_networkx_labels(G, pos, labels, font_size=font_size)
        
        # Add legend
        passage_patch = mpatches.Patch(color='lightblue', label='Passages')
        concept_patch = mpatches.Patch(color='lightcoral', label='Concepts')
        plt.legend(handles=[passage_patch, concept_patch], loc='upper right')
        
        plt.title(f"TERAG Graph Visualization\n{len(passage_nodes)} Passages, {len(concept_nodes)} Concepts")
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plotly_interactive(
        self,
        save_path: Optional[str] = None,
        show_in_browser: bool = True,
        layout: str = "spring"
    ) -> Optional[go.Figure]:
        """
        Create interactive Plotly visualization
        
        Args:
            save_path: Path to save HTML file
            show_in_browser: Whether to display in browser
            layout: Layout algorithm
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required. Install with: pip install plotly")
            
        self._ensure_networkx_graph(include_edge_weights=True)
        G = self.networkx_graph
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        else:
            pos = nx.spring_layout(G)
        
        # Extract edge coordinates
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Separate node types
        passage_nodes = []
        concept_nodes = []
        passage_info = []
        concept_info = []
        
        for node, data in G.nodes(data=True):
            x, y = pos[node]
            
            if data.get('node_type') == 'passage':
                passage_nodes.append([x, y])
                content = data.get('content', 'No content')
                passage_info.append(f"Passage {data.get('chunk_index', '')}<br>{content}")
            else:
                concept_nodes.append([x, y])
                concept_text = data.get('concept_text', str(node))
                concept_type = data.get('concept_type', 'unknown')
                frequency = data.get('frequency', 0)
                concept_info.append(f"{concept_text}<br>Type: {concept_type}<br>Frequency: {frequency}")
        
        traces = [edge_trace]
        
        # Add passage nodes
        if passage_nodes:
            passage_x, passage_y = zip(*passage_nodes)
            passage_trace = go.Scatter(
                x=passage_x, y=passage_y,
                mode='markers',
                hoverinfo='text',
                text=passage_info,
                marker=dict(
                    size=10,
                    color='lightblue',
                    line=dict(width=1, color='black')
                ),
                name='Passages'
            )
            traces.append(passage_trace)
        
        # Add concept nodes
        if concept_nodes:
            concept_x, concept_y = zip(*concept_nodes)
            concept_trace = go.Scatter(
                x=concept_x, y=concept_y,
                mode='markers',
                hoverinfo='text',
                text=concept_info,
                marker=dict(
                    size=6,
                    color='lightcoral',
                    line=dict(width=1, color='black')
                ),
                name='Concepts'
            )
            traces.append(concept_trace)
        
        # Create figure
        fig = go.Figure(data=traces,
                       layout=go.Layout(
                           title=f'Interactive TERAG Graph<br>{G.number_of_nodes()} nodes, {G.number_of_edges()} edges',
                           titlefont_size=16,
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Click and drag to pan, scroll to zoom",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        # Save and/or display
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")
        
        if show_in_browser:
            fig.show()
        
        return fig
    
    def pyvis_network(
        self,
        save_path: str = "terag_network.html",
        height: str = "600px",
        width: str = "100%",
        physics: bool = True
    ) -> None:
        """
        Create Pyvis network visualization
        
        Args:
            save_path: Path to save HTML file
            height: Network height
            width: Network width
            physics: Enable physics simulation
        """
        if not PYVIS_AVAILABLE:
            raise ImportError("Pyvis is required. Install with: pip install pyvis")
            
        self._ensure_networkx_graph()
        G = self.networkx_graph
        
        # Create Pyvis network
        net = Network(height=height, width=width, bgcolor="#ffffff", font_color="black")
        
        # Configure physics
        if physics:
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100}
              }
            }
            """)
        
        # Add nodes
        for node, data in G.nodes(data=True):
            if data.get('node_type') == 'passage':
                title = f"Passage {data.get('chunk_index', '')}\n{data.get('content', '')[:200]}..."
                net.add_node(node, label=f"P{data.get('chunk_index', '')}", 
                           color="lightblue", title=title, size=15)
            else:
                concept_text = data.get('concept_text', str(node))
                frequency = data.get('frequency', 0)
                title = f"{concept_text}\nType: {data.get('concept_type', '')}\nFrequency: {frequency}"
                net.add_node(node, label=concept_text[:15], 
                           color="lightcoral", title=title, size=8)
        
        # Add edges
        for edge in G.edges():
            net.add_edge(edge[0], edge[1])
        
        # Save
        net.save_graph(save_path)
        print(f"Pyvis network saved to {save_path}")
    
    def export_formats(self, base_path: str = "terag_graph") -> Dict[str, str]:
        """
        Export graph to various formats
        
        Args:
            base_path: Base filename (without extension)
            
        Returns:
            Dictionary of format -> file path
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for export")
            
        self._ensure_networkx_graph()
        G = self.networkx_graph
        
        exported_files = {}
        
        # GraphML (XML-based, preserves attributes)
        try:
            graphml_path = f"{base_path}.graphml"
            nx.write_graphml(G, graphml_path)
            exported_files['GraphML'] = graphml_path
        except Exception as e:
            warnings.warn(f"Failed to export GraphML: {e}")
        
        # GML (simpler format)
        try:
            gml_path = f"{base_path}.gml"
            nx.write_gml(G, gml_path)
            exported_files['GML'] = gml_path
        except Exception as e:
            warnings.warn(f"Failed to export GML: {e}")
        
        # GEXF (Gephi format)
        try:
            gexf_path = f"{base_path}.gexf"
            nx.write_gexf(G, gexf_path)
            exported_files['GEXF'] = gexf_path
        except Exception as e:
            warnings.warn(f"Failed to export GEXF: {e}")
        
        # Edge list
        try:
            edgelist_path = f"{base_path}.edgelist"
            nx.write_edgelist(G, edgelist_path)
            exported_files['EdgeList'] = edgelist_path
        except Exception as e:
            warnings.warn(f"Failed to export EdgeList: {e}")
        
        print(f"Exported {len(exported_files)} formats: {list(exported_files.keys())}")
        return exported_files
    
    def print_graph_summary(self) -> None:
        """Print summary statistics of the graph"""
        stats = self.terag_graph.get_statistics()
        nx_stats = self.terag_graph.get_networkx_statistics()
        
        print("\n" + "="*50)
        print("TERAG GRAPH SUMMARY")
        print("="*50)
        
        print(f"Passages: {stats['num_passages']}")
        print(f"Concepts: {stats['num_concepts']}")
        print(f"Edges: {stats['num_edges']}")
        print(f"Avg concepts per passage: {stats['avg_concepts_per_passage']:.2f}")
        print(f"Avg passages per concept: {stats['avg_passages_per_concept']:.2f}")
        
        if nx_stats.get('error') is None:
            print(f"Graph density: {nx_stats['density']:.4f}")
            print(f"Connected components: {nx_stats['num_connected_components']}")
            print(f"Is bipartite: {nx_stats['is_bipartite']}")
            
            if 'top_nodes_by_degree' in nx_stats:
                print("\nTop nodes by degree:")
                for node, degree in nx_stats['top_nodes_by_degree']:
                    node_type = "Passage" if node.startswith('passage_') else "Concept"
                    print(f"  {node} ({node_type}): {degree} connections")
        
        # Concept frequency distribution
        if stats.get('concept_frequency_distribution'):
            freq_dist = stats['concept_frequency_distribution']
            print(f"\nConcept frequency distribution:")
            print(f"  Min: {freq_dist.get('min', 'N/A')}")
            print(f"  Max: {freq_dist.get('max', 'N/A')}")
            print(f"  Mean: {freq_dist.get('mean', 'N/A'):.2f}")
            print(f"  Median: {freq_dist.get('median', 'N/A'):.2f}")


def visualize_terag_graph(
    terag_graph: TERAGGraph,
    output_dir: str = "visualizations",
    formats: List[str] = ["matplotlib", "plotly", "pyvis"],
    export_data: bool = True
) -> Dict[str, str]:
    """
    Create all visualizations for a TERAG graph
    
    Args:
        terag_graph: TERAGGraph instance
        output_dir: Directory to save outputs
        formats: List of visualization formats to create
        export_data: Whether to export graph data files
        
    Returns:
        Dictionary of created files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    visualizer = GraphVisualizer(terag_graph)
    created_files = {}
    
    # Print summary
    visualizer.print_graph_summary()
    
    # Create visualizations
    if "matplotlib" in formats:
        try:
            matplotlib_path = output_path / "terag_graph_static.png"
            visualizer.matplotlib_plot(save_path=str(matplotlib_path), show_labels=True)
            created_files['matplotlib'] = str(matplotlib_path)
        except Exception as e:
            print(f"Failed to create matplotlib visualization: {e}")
    
    if "plotly" in formats:
        try:
            plotly_path = output_path / "terag_graph_interactive.html"
            visualizer.plotly_interactive(save_path=str(plotly_path), show_in_browser=False)
            created_files['plotly'] = str(plotly_path)
        except Exception as e:
            print(f"Failed to create plotly visualization: {e}")
    
    if "pyvis" in formats:
        try:
            pyvis_path = output_path / "terag_graph_network.html"
            visualizer.pyvis_network(save_path=str(pyvis_path))
            created_files['pyvis'] = str(pyvis_path)
        except Exception as e:
            print(f"Failed to create pyvis visualization: {e}")
    
    # Export data formats
    if export_data:
        try:
            export_files = visualizer.export_formats(str(output_path / "terag_graph"))
            created_files.update(export_files)
        except Exception as e:
            print(f"Failed to export data files: {e}")
    
    print(f"\n\nCreated {len(created_files)} files in {output_dir}/")
    for format_name, file_path in created_files.items():
        print(f"  {format_name}: {file_path}")
    
    return created_files


if __name__ == "__main__":
    # Test visualization with example data
    from graph_builder import GraphBuilder
    
    def dummy_extract_concepts(text: str):
        """Dummy concept extractor for testing"""
        words = text.split()
        named_entities = [w for w in words if w[0].isupper() and len(w) > 2]
        doc_concepts = [w.lower() for w in words if len(w) > 8]
        return named_entities, doc_concepts
    
    # Test chunks
    test_chunks = [
        {"content": "Apple Inc announced revenue growth in California market technology."},
        {"content": "Microsoft Corporation expanded operations in Washington development."},
        {"content": "Apple and Microsoft compete in technology sector innovation."},
        {"content": "California technology companies focus on artificial intelligence."},
        {"content": "Washington development initiatives support innovation technology."}
    ]
    
    # Build graph
    builder = GraphBuilder(min_concept_freq=1)
    graph = builder.build_graph_from_chunks(test_chunks, dummy_extract_concepts)
    
    # Create visualizations
    visualize_terag_graph(graph, formats=["matplotlib", "plotly", "pyvis"])