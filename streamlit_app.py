import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import time
import random

# ---------------------------
# Page Config & Custom CSS
# ---------------------------
st.set_page_config(page_title="TSP Game", layout="wide")

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
h1, h2, h3, h4, h5 {
    color: #ffffff;
    text-shadow: 2px 2px 4px black;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: bold;
    transition: 0.3s;
    border: none;
}
.stButton>button:hover {
    background-color: #45a049;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.game-stats {
    background: rgba(255, 255, 255, 0.9);
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
.city-button {
    background-color: #2196F3;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    margin: 5px;
    font-size: 14px;
}
.visited-city {
    background-color: #FF9800;
}
.current-city {
    background-color: #F44336;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---------------------------
# Session State Initialization
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = 1

if "game_state" not in st.session_state:
    st.session_state.game_state = {
        "graph": None,
        "current_path": [],
        "current_cost": 0,
        "visited_cities": set(),
        "game_over": False,
        "start_city": 0,
        "cities_names": [],
        "best_known_cost": float('inf'),
        "player_attempts": [],
        "ai_solutions": {},  # Store AI algorithm results
        "show_hints": False,
        "algorithm_comparison": False
    }

def reset_game():
    st.session_state.game_state = {
        "graph": st.session_state.game_state["graph"],
        "current_path": [],
        "current_cost": 0,
        "visited_cities": set(),
        "game_over": False,
        "start_city": 0,
        "cities_names": st.session_state.game_state["cities_names"],
        "best_known_cost": st.session_state.game_state["best_known_cost"],
        "player_attempts": st.session_state.game_state["player_attempts"],
        "ai_solutions": st.session_state.game_state["ai_solutions"],
        "show_hints": False,
        "algorithm_comparison": st.session_state.game_state.get("algorithm_comparison", False)
    }

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

def go_to_page(page_num):
    st.session_state.page = page_num

# ---------------------------
# Graph Visualization
# ---------------------------
def visualize_game_graph(graph, current_path=None, highlight_available=None, title="Your City Network"):
    if not graph or len(graph.nodes) == 0:
        return
        
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 8))
    
    # Draw all edges with weights
    nx.draw_networkx_edges(graph, pos, alpha=0.5, width=1)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=10)
    
    # Color nodes based on game state
    node_colors = []
    for node in graph.nodes():
        if current_path and len(current_path) > 0 and node == current_path[-1]:
            node_colors.append('red')  # Current city
        elif node in st.session_state.game_state["visited_cities"]:
            node_colors.append('orange')  # Visited cities
        elif highlight_available and node in highlight_available:
            node_colors.append('lightgreen')  # Available cities
        else:
            node_colors.append('lightblue')  # Unvisited cities
    
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1000, alpha=0.8)
    
    # Draw city names
    city_names = st.session_state.game_state["cities_names"]
    labels = {i: city_names[i] if i < len(city_names) else str(i) for i in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels, font_size=12, font_weight='bold')
    
    # Draw current path
    if current_path and len(current_path) > 1:
        path_edges = [(current_path[i], current_path[i+1]) for i in range(len(current_path)-1)]
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=4, alpha=0.7)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    st.pyplot(plt.gcf())
    plt.close()

# ---------------------------
# TSP Algorithms
# ---------------------------
def bfs_tsp(graph, start):
    """Breadth-First Search for TSP"""
    from collections import deque
    queue = deque([(start, [start], 0)])
    best_path, best_cost = None, float('inf')
    nodes_explored = 0
    
    while queue:
        city, path, cost = queue.popleft()
        nodes_explored += 1
        
        if len(path) == len(graph.nodes):
            if graph.has_edge(path[-1], start):
                total_cost = cost + graph[path[-1]][start]['weight']
                if total_cost < best_cost:
                    best_path, best_cost = path + [start], total_cost
        else:
            for neighbor in graph.neighbors(city):
                if neighbor not in path:
                    new_cost = cost + graph[city][neighbor]['weight']
                    queue.append((neighbor, path + [neighbor], new_cost))
    
    return best_path, best_cost, nodes_explored

def dfs_tsp(graph, start):
    """Depth-First Search for TSP"""
    stack = [(start, [start], 0)]
    best_path, best_cost = None, float('inf')
    nodes_explored = 0
    
    while stack:
        city, path, cost = stack.pop()
        nodes_explored += 1
        
        if len(path) == len(graph.nodes):
            if graph.has_edge(path[-1], start):
                total_cost = cost + graph[path[-1]][start]['weight']
                if total_cost < best_cost:
                    best_path, best_cost = path + [start], total_cost
        else:
            for neighbor in graph.neighbors(city):
                if neighbor not in path:
                    new_cost = cost + graph[city][neighbor]['weight']
                    stack.append((neighbor, path + [neighbor], new_cost))
    
    return best_path, best_cost, nodes_explored

def ucs_tsp(graph, start):
    """Uniform Cost Search for TSP"""
    import heapq
    pq = [(0, start, [start])]
    best_path, best_cost = None, float('inf')
    nodes_explored = 0
    
    while pq:
        cost, city, path = heapq.heappop(pq)
        nodes_explored += 1
        
        if len(path) == len(graph.nodes):
            if graph.has_edge(path[-1], start):
                total_cost = cost + graph[path[-1]][start]['weight']
                if total_cost < best_cost:
                    best_path, best_cost = path + [start], total_cost
        else:
            for neighbor in graph.neighbors(city):
                if neighbor not in path:
                    new_cost = cost + graph[city][neighbor]['weight']
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))
    
    return best_path, best_cost, nodes_explored

def heuristic_tsp(graph, path, start):
    """Heuristic function for A* - minimum cost to complete tour"""
    remaining = set(graph.nodes) - set(path)
    if not remaining:
        if graph.has_edge(path[-1], start):
            return graph[path[-1]][start]['weight']
        else:
            return float('inf')
    
    # Simple heuristic: minimum edge from current city to any remaining city
    current = path[-1]
    min_cost = float('inf')
    for city in remaining:
        if graph.has_edge(current, city):
            min_cost = min(min_cost, graph[current][city]['weight'])
    
    return min_cost if min_cost != float('inf') else 0

def astar_tsp(graph, start):
    """A* Search for TSP"""
    import heapq
    pq = [(0, start, [start])]
    best_path, best_cost = None, float('inf')
    nodes_explored = 0
    
    while pq:
        f_cost, city, path = heapq.heappop(pq)
        nodes_explored += 1
        
        # Calculate actual cost so far
        actual_cost = 0
        for i in range(len(path) - 1):
            actual_cost += graph[path[i]][path[i+1]]['weight']
        
        if len(path) == len(graph.nodes):
            if graph.has_edge(path[-1], start):
                total_cost = actual_cost + graph[path[-1]][start]['weight']
                if total_cost < best_cost:
                    best_path, best_cost = path + [start], total_cost
        else:
            for neighbor in graph.neighbors(city):
                if neighbor not in path:
                    g_cost = actual_cost + graph[city][neighbor]['weight']
                    h_cost = heuristic_tsp(graph, path + [neighbor], start)
                    f_cost = g_cost + h_cost
                    heapq.heappush(pq, (f_cost, neighbor, path + [neighbor]))
    
    return best_path, best_cost, nodes_explored

def run_algorithm(graph, start_city, algorithm_name):
    """Run the specified algorithm and return results with timing"""
    start_time = time.time()
    
    if algorithm_name == "BFS":
        path, cost, nodes = bfs_tsp(graph, start_city)
    elif algorithm_name == "DFS":
        path, cost, nodes = dfs_tsp(graph, start_city)
    elif algorithm_name == "UCS":
        path, cost, nodes = ucs_tsp(graph, start_city)
    elif algorithm_name == "A*":
        path, cost, nodes = astar_tsp(graph, start_city)
    else:
        return None, float('inf'), 0, 0
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return path, cost, nodes, execution_time

# ---------------------------
# Game Logic
# ---------------------------
def calculate_optimal_solution(graph, start_city):
    """Calculate the optimal TSP solution using brute force for comparison"""
    from itertools import permutations
    
    nodes = list(graph.nodes())
    nodes.remove(start_city)
    
    min_cost = float('inf')
    best_path = None
    
    for perm in permutations(nodes):
        path = [start_city] + list(perm) + [start_city]
        cost = 0
        valid = True
        
        for i in range(len(path)-1):
            if graph.has_edge(path[i], path[i+1]):
                cost += graph[path[i]][path[i+1]]['weight']
            else:
                valid = False
                break
                
        if valid and cost < min_cost:
            min_cost = cost
            best_path = path
    
    return best_path, min_cost

def move_to_city(city):
    """Handle player's move to a city"""
    game = st.session_state.game_state
    
    if game["game_over"]:
        return
        
    current_city = game["current_path"][-1] if game["current_path"] else game["start_city"]
    
    # Add to path and calculate cost
    if not game["current_path"]:
        game["current_path"] = [game["start_city"]]
        current_city = game["start_city"]
    
    # Check if the edge exists before trying to access it
    if city != current_city and game["graph"].has_edge(current_city, city):
        travel_cost = game["graph"][current_city][city]['weight']
        game["current_path"].append(city)
        game["current_cost"] += travel_cost
        game["visited_cities"].add(city)
        
        # Check if game is complete
        if len(game["visited_cities"]) == len(game["graph"].nodes()) - 1:  # -1 because start city is not in visited_cities
            # Check if we can return to start
            if game["graph"].has_edge(city, game["start_city"]):
                return_cost = game["graph"][city][game["start_city"]]['weight']
                final_cost = game["current_cost"] + return_cost
                game["player_attempts"].append(final_cost)
                
                st.success(f"🎉 Journey Complete! Total Cost: {final_cost}")
                if final_cost <= game["best_known_cost"]:
                    st.balloons()
                    st.success("🏆 New Best Score!")
                    game["best_known_cost"] = final_cost
            else:
                st.warning("⚠️ Cannot complete journey - no path back to start city!")
    else:
        st.error("⚠️ Invalid move - no direct path to that city!")

# ---------------------------
# Pages
# ---------------------------

if st.session_state.page == 1:
    # Welcome Page
    st.title("🌍 TSP Adventure Game")
    st.markdown("""
    ### Welcome, Travel Planner! ✈️
    
    **Your Mission:** Plan the most efficient route to visit all cities and return home!
    
    **How to Play:**
    1. 🏗️ **Build Your World** - Choose number of cities and set distances
    2. 🎮 **Play the Game** - Click cities to travel and find the shortest route
    3. 🏆 **Beat the Best** - Try to find the optimal solution!
    
    **Game Features:**
    - 🎯 Interactive city selection
    - 📊 Real-time cost tracking  
    - 🏅 Score comparison with optimal solution
    - 🔄 Multiple attempts to improve your route
    
    Ready to become the ultimate travel optimizer?
    """)
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.button("🚀 Start Adventure", on_click=next_page, key="start_btn")

elif st.session_state.page == 2:
    # Game Setup Page
    st.header("🏗️ Build Your City Network")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Network Configuration")
        n_cities = st.slider("Number of Cities", 3, 8, 5)
        
        # Generate random city names
        city_names = ["New York", "London", "Tokyo", "Paris", "Sydney", "Mumbai", "Cairo", "Berlin", "Rio", "Toronto"]
        selected_names = random.sample(city_names, n_cities)
        
        st.subheader("Your Cities")
        for i, name in enumerate(selected_names):
            st.write(f"🏙️ **City {i}:** {name}")
        
        # Option to randomize distances or set manually
        setup_mode = st.radio("Distance Setup", ["Random Distances", "Manual Setup"])
        
        if setup_mode == "Random Distances":
            if st.button("🎲 Generate Random Network"):
                # Create graph with random weights
                G = nx.complete_graph(n_cities)
                for i, j in G.edges():
                    weight = random.randint(10, 100)
                    G[i][j]['weight'] = weight
                
                st.session_state.game_state["graph"] = G
                st.session_state.game_state["cities_names"] = selected_names
                
                # Calculate optimal solution and run all algorithms
                optimal_path, optimal_cost = calculate_optimal_solution(G, 0)
                st.session_state.game_state["best_known_cost"] = optimal_cost
                
                # Run all AI algorithms for comparison
                st.session_state.game_state["ai_solutions"] = {}
                algorithms = ["BFS", "DFS", "UCS", "A*"]
                for alg in algorithms:
                    path, cost, nodes, exec_time = run_algorithm(G, 0, alg)
                    st.session_state.game_state["ai_solutions"][alg] = {
                        "path": path,
                        "cost": cost,
                        "nodes_explored": nodes,
                        "execution_time": exec_time
                    }
        else:
            st.subheader("Set Distances Manually")
            if st.session_state.game_state["graph"] is None or len(st.session_state.game_state["graph"].nodes()) != n_cities:
                weights = {}
                G = nx.complete_graph(n_cities)
                
                for i in range(n_cities):
                    for j in range(i+1, n_cities):
                        default_weight = random.randint(15, 80)
                        weights[(i, j)] = st.slider(
                            f"{selected_names[i]} ↔ {selected_names[j]}", 
                            10, 200, default_weight, key=f"weight_{i}_{j}"
                        )
                
                # Create graph
                for (i, j), w in weights.items():
                    G[i][j]['weight'] = w
                
                st.session_state.game_state["graph"] = G
                st.session_state.game_state["cities_names"] = selected_names
                
                # Calculate optimal solution and run all algorithms
                optimal_path, optimal_cost = calculate_optimal_solution(G, 0)
                st.session_state.game_state["best_known_cost"] = optimal_cost
                
                # Run all AI algorithms for comparison
                st.session_state.game_state["ai_solutions"] = {}
                algorithms = ["BFS", "DFS", "UCS", "A*"]
                for alg in algorithms:
                    path, cost, nodes, exec_time = run_algorithm(G, 0, alg)
                    st.session_state.game_state["ai_solutions"][alg] = {
                        "path": path,
                        "cost": cost,
                        "nodes_explored": nodes,
                        "execution_time": exec_time
                    }
    
    with col2:
        if st.session_state.game_state["graph"] is not None:
            st.subheader("Network Preview")
            visualize_game_graph(st.session_state.game_state["graph"], title="Your City Network")
            st.success(f"✅ Network Ready! Optimal route cost: {st.session_state.game_state['best_known_cost']}")
            
            # Show algorithm comparison
            if st.session_state.game_state["ai_solutions"]:
                with st.expander("🤖 AI Algorithm Comparison"):
                    comparison_data = []
                    for alg, results in st.session_state.game_state["ai_solutions"].items():
                        comparison_data.append({
                            "Algorithm": alg,
                            "Best Cost": results["cost"],
                            "Nodes Explored": results["nodes_explored"],
                            "Time (ms)": f"{results['execution_time']*1000:.2f}",
                            "Optimal?": "✅" if results["cost"] == st.session_state.game_state["best_known_cost"] else "❌"
                        })
                    
                    st.dataframe(comparison_data, use_container_width=True)
                    
                    # Show which algorithms found optimal solution
                    optimal_algorithms = [alg for alg, results in st.session_state.game_state["ai_solutions"].items() 
                                        if results["cost"] == st.session_state.game_state["best_known_cost"]]
                    if optimal_algorithms:
                        st.info(f"🏆 **Algorithms that found optimal solution:** {', '.join(optimal_algorithms)}")
    
    # Navigation
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.button("⬅️ Back", on_click=prev_page)
    with col3:
        if st.session_state.game_state["graph"] is not None:
            st.button("Play Game ➡️", on_click=next_page)

elif st.session_state.page == 3:
    # Main Game Page
    game = st.session_state.game_state
    
    st.header("🎮 TSP Adventure - Plan Your Route!")
    
    if game["graph"] is None:
        st.error("Please set up your city network first!")
        st.button("⬅️ Back to Setup", on_click=prev_page)
        st.stop()
    
    # Game Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="game-stats">', unsafe_allow_html=True)
        st.metric("Current Cost", game["current_cost"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="game-stats">', unsafe_allow_html=True)
        cities_visited = len(game["visited_cities"]) + (1 if game["current_path"] else 0)
        total_cities = len(game["graph"].nodes())
        st.metric("Cities Visited", f"{cities_visited}/{total_cities}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="game-stats">', unsafe_allow_html=True)
        st.metric("Best Known", game["best_known_cost"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="game-stats">', unsafe_allow_html=True)
        st.metric("Attempts", len(game["player_attempts"]))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Current Status
    current_city = game["current_path"][-1] if game["current_path"] else game["start_city"]
    current_city_name = game["cities_names"][current_city]
    st.info(f"📍 Currently at: **{current_city_name}**")
    
    # Game Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Visualize current state
        available_cities = []
        if not game["game_over"]:
            for city in game["graph"].nodes():
                if city not in game["visited_cities"] and city != current_city and city != game["start_city"]:
                    available_cities.append(city)
        
        visualize_game_graph(
            game["graph"], 
            game["current_path"], 
            available_cities,
            "Click a city below to travel there!"
        )
    
    with col2:
        st.subheader("🎯 Choose Next City")
        
        # AI Assistant Panel
        st.markdown("### 🤖 AI Assistant")
        
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("💡 Get Hint", key="get_hint"):
                game["show_hints"] = not game.get("show_hints", False)
        
        with col2b:
            if st.button("🏆 Show AI Solutions", key="show_ai"):
                game["algorithm_comparison"] = not game.get("algorithm_comparison", False)
        
        # Show hints if enabled
        if game.get("show_hints", False) and game["ai_solutions"]:
            with st.expander("💡 AI Hints", expanded=True):
                # Show next best move from different algorithms
                current_city = game["current_path"][-1] if game["current_path"] else game["start_city"]
                current_step = len(game["current_path"]) if game["current_path"] else 0
                
                hint_suggestions = {}
                for alg, results in game["ai_solutions"].items():
                    if results["path"] and current_step < len(results["path"]) - 1:
                        next_city = results["path"][current_step + 1] if current_step + 1 < len(results["path"]) else None
                        if next_city is not None and next_city != game["start_city"]:  # Don't suggest going home early
                            city_name = game["cities_names"][next_city]
                            if next_city not in hint_suggestions:
                                hint_suggestions[next_city] = []
                            hint_suggestions[next_city].append(alg)
                
                if hint_suggestions:
                    st.write("**Recommended next cities:**")
                    for city, algorithms in hint_suggestions.items():
                        city_name = game["cities_names"][city]
                        alg_list = ", ".join(algorithms)
                        if game["graph"].has_edge(current_city, city):
                            cost = game["graph"][current_city][city]['weight']
                            st.write(f"• **{city_name}** (Cost: {cost}) - Suggested by: {alg_list}")
                else:
                    st.write("No specific recommendations at this step.")
        
        # Show algorithm comparison if enabled
        if game.get("algorithm_comparison", False) and game["ai_solutions"]:
            with st.expander("🏆 AI Algorithm Solutions", expanded=True):
                for alg, results in game["ai_solutions"].items():
                    if results["path"]:
                        path_str = " → ".join([game["cities_names"][city] for city in results["path"]])
                        efficiency = "🏆 Optimal" if results["cost"] == game["best_known_cost"] else f"🔄 +{results['cost'] - game['best_known_cost']}"
                        st.write(f"**{alg}:** {efficiency}")
                        st.write(f"Path: {path_str}")
                        st.write(f"Cost: {results['cost']} | Nodes: {results['nodes_explored']} | Time: {results['execution_time']*1000:.1f}ms")
                        st.markdown("---")
        
        st.markdown("### 🚀 Your Move")
        
        if not game["current_path"]:
            # Starting the journey
            start_city_name = game["cities_names"][game["start_city"]]
            st.write(f"🏠 Starting from: **{start_city_name}**")
            if st.button(f"Begin Journey from {start_city_name}", key="start_journey"):
                game["current_path"] = [game["start_city"]]
                st.rerun()
        
        elif len(game["visited_cities"]) < len(game["graph"].nodes()) - 1:
            # Continue journey
            st.write("Available destinations:")
            for city in game["graph"].nodes():
                if city not in game["visited_cities"] and city != current_city and city != game["start_city"]:
                    # Check if there's a direct path to this city
                    if game["graph"].has_edge(current_city, city):
                        city_name = game["cities_names"][city]
                        travel_cost = game["graph"][current_city][city]['weight']
                        
                        if st.button(f"✈️ {city_name} (Cost: {travel_cost})", key=f"move_{city}"):
                            move_to_city(city)
                            st.rerun()
            
            # Check if no moves are available
            available_moves = [city for city in game["graph"].nodes() 
                             if city not in game["visited_cities"] 
                             and city != current_city 
                             and city != game["start_city"]
                             and game["graph"].has_edge(current_city, city)]
            
            if not available_moves:
                st.error("⚠️ No available moves from current city! Game stuck.")
                if st.button("🔄 Reset Game", key="stuck_reset"):
                    reset_game()
                    st.rerun()
        
        else:
            # Return home
            st.write("🏠 All cities visited! Return home:")
            start_city_name = game["cities_names"][game["start_city"]]
            
            # Check if there's a direct edge back to start
            if game["graph"].has_edge(current_city, game["start_city"]):
                return_cost = game["graph"][current_city][game["start_city"]]['weight']
                
                if st.button(f"🏠 Return to {start_city_name} (Cost: {return_cost})", key="return_home"):
                    game["current_path"].append(game["start_city"])
                    final_cost = game["current_cost"] + return_cost
                    game["current_cost"] = final_cost
                    game["player_attempts"].append(final_cost)
                    game["game_over"] = True
                    
                    st.success(f"🎉 Journey Complete! Final Cost: {final_cost}")
                    if final_cost <= game["best_known_cost"]:
                        st.balloons()
                        st.success("🏆 Optimal Route Found!")
                    st.rerun()
            else:
                st.error(f"⚠️ No direct path back to {start_city_name}! This shouldn't happen in a complete graph.")
                st.write("Available paths back:")
                for city in game["graph"].neighbors(current_city):
                    if city == game["start_city"]:
                        continue
                    city_name = game["cities_names"][city]
                    cost_to_city = game["graph"][current_city][city]['weight']
                    if game["graph"].has_edge(city, game["start_city"]):
                        cost_to_start = game["graph"][city][game["start_city"]]['weight']
                        total_cost = cost_to_city + cost_to_start
                        st.write(f"Via {city_name}: {cost_to_city} + {cost_to_start} = {total_cost}")
        
        # Game Controls
        st.markdown("---")
        st.markdown("### ⚙️ Game Controls")
        
        control_col1, control_col2 = st.columns(2)
        
        with control_col1:
            if st.button("🔄 Reset Game", key="reset"):
                reset_game()
                st.rerun()
            
            if st.button("🎲 Random Move", key="random_move"):
                if game["current_path"] and len(game["visited_cities"]) < len(game["graph"].nodes()) - 1:
                    current_city = game["current_path"][-1]
                    available_cities = [city for city in game["graph"].nodes() 
                                      if city not in game["visited_cities"] 
                                      and city != current_city 
                                      and city != game["start_city"]
                                      and game["graph"].has_edge(current_city, city)]
                    if available_cities:
                        random_city = random.choice(available_cities)
                        move_to_city(random_city)
                        st.rerun()
        
        with control_col2:
            if st.button("⚙️ New Network", key="new_network"):
                st.session_state.game_state = {
                    "graph": None,
                    "current_path": [],
                    "current_cost": 0,
                    "visited_cities": set(),
                    "game_over": False,
                    "start_city": 0,
                    "cities_names": [],
                    "best_known_cost": float('inf'),
                    "player_attempts": [],
                    "ai_solutions": {},
                    "show_hints": False,
                    "algorithm_comparison": False
                }
                go_to_page(2)
                st.rerun()
            
            # AI Challenge mode
            if st.button("🤖 AI Challenge", key="ai_challenge"):
                if game["ai_solutions"]:
                    # Show challenge results
                    st.balloons()
                    player_best = min(game["player_attempts"]) if game["player_attempts"] else float('inf')
                    ai_best = min([results["cost"] for results in game["ai_solutions"].values()])
                    
                    if player_best <= ai_best:
                        st.success("🏆 You beat the AI algorithms!")
                    else:
                        st.info(f"🤖 AI wins! Best AI: {ai_best}, Your best: {player_best}")
        
        # Advanced Features
        with st.expander("🔬 Advanced Features"):
            st.markdown("**Algorithm Details:**")
            if game["ai_solutions"]:
                for alg, results in game["ai_solutions"].items():
                    efficiency_score = (game["best_known_cost"] / results["cost"]) * 100 if results["cost"] > 0 else 0
                    st.write(f"• **{alg}**: {efficiency_score:.1f}% efficiency")
            
            if st.button("📊 Export Results", key="export"):
                # Create downloadable results summary
                results_summary = {
                    "Network": f"{len(game['graph'].nodes())} cities",
                    "Optimal Cost": game["best_known_cost"],
                    "Player Attempts": game["player_attempts"],
                    "Player Best": min(game["player_attempts"]) if game["player_attempts"] else "N/A",
                    "AI Results": {alg: results["cost"] for alg, results in game["ai_solutions"].items()}
                }
                st.json(results_summary)
    
    # Score History & AI Comparison
    if game["player_attempts"]:
        st.subheader("📈 Performance Dashboard")
        
        # Create performance comparison
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.write("**Your Performance:**")
            attempts_df = {
                "Attempt": list(range(1, len(game["player_attempts"]) + 1)),
                "Cost": game["player_attempts"],
                "Difference from Optimal": [cost - game["best_known_cost"] for cost in game["player_attempts"]]
            }
            st.dataframe(attempts_df, use_container_width=True)
            
            best_attempt = min(game["player_attempts"])
            st.metric("Your Best Score", best_attempt, 
                     delta=best_attempt - game["best_known_cost"], 
                     delta_color="inverse")
        
        with perf_col2:
            if game["ai_solutions"]:
                st.write("**AI Algorithms vs You:**")
                comparison_data = []
                player_best = min(game["player_attempts"])
                
                for alg, results in game["ai_solutions"].items():
                    vs_player = results["cost"] - player_best
                    vs_optimal = results["cost"] - game["best_known_cost"]
                    comparison_data.append({
                        "Algorithm": alg,
                        "Cost": results["cost"],
                        "vs You": f"{vs_player:+d}",
                        "vs Optimal": f"{vs_optimal:+d}",
                        "Winner": "🤖 AI" if results["cost"] < player_best else "👤 You" if results["cost"] > player_best else "🤝 Tie"
                    })
                
                st.dataframe(comparison_data, use_container_width=True)
                
                # Overall challenge result
                ai_wins = sum(1 for data in comparison_data if data["Winner"] == "🤖 AI")
                player_wins = sum(1 for data in comparison_data if data["Winner"] == "👤 You")
                
                if player_wins > ai_wins:
                    st.success(f"🏆 You're winning! {player_wins} vs {ai_wins}")
                elif ai_wins > player_wins:
                    st.info(f"🤖 AI is ahead: {ai_wins} vs {player_wins}")
                else:
                    st.warning("🤝 It's a tie! Keep playing to break it!")
    
    elif game["ai_solutions"]:
        st.subheader("🤖 AI Algorithm Performance")
        st.write("See how different algorithms performed on your network:")
        
        for alg, results in game["ai_solutions"].items():
            with st.expander(f"{alg} Algorithm Results"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Cost", results["cost"])
                with col2:
                    st.metric("Nodes Explored", results["nodes_explored"])
                with col3:
                    st.metric("Execution Time", f"{results['execution_time']*1000:.2f}ms")
                
                if results["path"]:
                    path_names = [game["cities_names"][city] for city in results["path"]]
                    st.write(f"**Route:** {' → '.join(path_names)}")
                
                # Algorithm-specific insights
                if alg == "BFS":
                    st.info("🔍 BFS explores all possible paths level by level. Guaranteed to find optimal solution but can be slow.")
                elif alg == "DFS":
                    st.info("🔍 DFS goes deep into each path before backtracking. Fast but may not find optimal solution first.")
                elif alg == "UCS":
                    st.info("🔍 UCS always expands the lowest-cost path first. Guaranteed optimal solution.")
                elif alg == "A*":
                    st.info("🔍 A* uses heuristics to guide search toward the goal. Usually fastest while maintaining optimality.")

# Footer
st.markdown("---")
st.markdown("🎮 **TSP Adventure Game** - Master the art of efficient travel planning!")
