import streamlit as st
from streamlit_folium import st_folium 

from src.geo import get_graph, resolve_both 
from src.routing import load_scored_graph, add_safe_costs, get_routes
from src.ranking import rank_and_explain, build_map

st.set_page_config(page_title="Safe Route Cycling Assistant", layout="wide")
st.title("Safe Route Cycling Assistant")
st.write("Find the safest routes for your next biking trip.")

@st.cache_resource
def get_scored_graph(): 
    G = load_scored_graph()
    add_safe_costs(G)
    return G

G = get_scored_graph()

col1, col2 = st.columns(2)
with col1: 
    start_text = st.text_input("Start", "San Jose State University, San Jose, CA")
with col2: 
    end_text = st.text_input("Destination", "Santana Row, San Jose, CA")
    
if st.button("Find Safe Route", type="primary"):
    try:
        with st.spinner("Finding routes..."):
            start_node, end_node = resolve_both(G, start_text, end_text)
            routes = get_routes(start_node, end_node, G)
            ranking = rank_and_explain(routes, G)
        st.session_state["routes"] = routes
        st.session_state["ranking"] = ranking
    except Exception as e: 
        st.error(f"Couldn't find a route: {e}")
        st.stop()
    
    #rank and explain the routes
    ranking = rank_and_explain(routes, G)
    
    #show the recommended route
if "routes" in st.session_state:
    routes = st.session_state["routes"]
    ranking = st.session_state["ranking"]
    
    metrics = ranking["metrics"]["safest"]
    
    distance_miles = metrics["total_length_km"] * 0.621371
    with st.container(border=True):
        st.success("**Recommended: Safest Route**")
        st.markdown(f"**Distance:** {distance_miles:.2f} mi")
        st.markdown(f"**Risk Score:** {metrics["avg_risk"]}")
        st.markdown(f"**Risk Segments:** {metrics["high_risk_segments"]} count")
        st.markdown(f"**Total Segments:** {metrics["total_edges"]} count")


    with st.container(border=True):
        st.success("**Why This Route?**")
        st.write(ranking["reasoning"])
    
    route_map = build_map(G, routes, ranking)
    st_folium(route_map, width=900, height=600)
        
