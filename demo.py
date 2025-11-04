import streamlit as st
import pandas as pd
import os
from groq import Groq
import numpy as np
import csv 
from datetime import datetime 
from dotenv import load_dotenv 

# ==============================
# 1. CONFIGURATION
# ==============================
# Load environment variables from the .env file.
load_dotenv() 

# NOTE: This key is loaded from the environment variable named GROQ_API_KEY
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# Using the model specified in the last user input
MODEL_NAME = "llama-3.1-8b-instant" 
LOG_FILE = "chat_log.csv" 

# Dataset files (ensure these are in same directory or adjust paths)
# For the app to run, these CSV files must be present in the execution environment.
DATA_FILES = {
    "city_day": "city_day.csv",
    "city_hour": "city_hour.csv",
    "station_day": "station_day.csv",
    "station_hour": "station_hour.csv",
    "stations": "stations.csv"
}

# ==============================
# 1.1 LOGGING FUNCTION
# ==============================
def log_chat_message(role, content):
    """Appends a message (user or assistant) to the chat log CSV."""
    try:
        file_exists = os.path.isfile(LOG_FILE)
        
        # Open in append mode ('a')
        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header only if the file does not exist
            if not file_exists:
                writer.writerow(['Timestamp', 'Role', 'Content'])
                
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Log the message
            writer.writerow([timestamp, role, content])
            
    except Exception as e:
        # Log a warning if the logging failed, but don't stop the main app
        st.warning(f"Could not log message to CSV: {e}")


# ==============================
# 2. DATA LOADING AND PREPARATION
# ==============================
@st.cache_data(show_spinner="Loading and preparing AQI datasets...")
def load_aqi_datasets(files):
    """
    Loads, processes, and prepares the AQI datasets for use as LLM context.
    It finds the latest available data point for each city and station.
    Crucially, it **samples** the data to create a small context for the LLM.
    Returns contexts and the actual dataframes for in-app analysis.
    """
    try:
        # Load datasets
        city_day = pd.read_csv(files["city_day"])
        station_day = pd.read_csv(files["station_day"])
        stations = pd.read_csv(files["stations"])
        
        # Convert dates
        city_day['Date'] = pd.to_datetime(city_day['Date'], errors='coerce')
        station_day['Date'] = pd.to_datetime(station_day['Date'], errors='coerce')

        # Get latest city data by finding the max date for each city
        latest_city = city_day.loc[city_day.groupby('City')['Date'].idxmax()]
        latest_city_data = latest_city[['City', 'Date', 'AQI', 'AQI_Bucket', 'PM2.5', 'PM10']].dropna(subset=['AQI'])
        latest_city_data['AQI'] = pd.to_numeric(latest_city_data['AQI'], errors='coerce').fillna(0)

        # Get latest station data by finding the max date for each station
        latest_station = station_day.loc[station_day.groupby('StationId')['Date'].idxmax()]
        # Merge with station metadata to get city and name
        latest_station = latest_station.merge(stations[['StationId', 'StationName', 'City']], on='StationId', how='left')
        latest_station_data = latest_station[['StationName', 'City', 'Date', 'AQI', 'AQI_Bucket', 'PM2.5', 'PM10']].dropna(subset=['AQI'])
        latest_station_data['AQI'] = pd.to_numeric(latest_station_data['AQI'], errors='coerce').fillna(0)


        # --- CONTEXT SAMPLING FOR LLM TOKEN REDUCTION ---
        
        # 1. City Context (Max 10 rows): Include the worst city + a small sample
        worst_city_row = latest_city_data.loc[latest_city_data['AQI'].idxmax()]
        
        # Get a sample of other cities, excluding the worst one (if applicable)
        sample_cities = latest_city_data[latest_city_data['City'] != worst_city_row['City']].sort_values(by='City').head(9)
        
        # Combine the worst city with the sample
        context_cities_df = pd.concat([worst_city_row.to_frame().T, sample_cities]).drop_duplicates(subset=['City']).reset_index(drop=True)
        
        # 2. Station Context (Max 15 rows): Take a small sample alphabetically
        context_stations_df = latest_station_data.sort_values(by='StationName').head(15).reset_index(drop=True)
        
        # Convert sampled dataframes to markdown tables for LLM context
        city_context = context_cities_df.to_markdown(index=False)
        station_context = context_stations_df.to_markdown(index=False)
        # -----------------------------------------------

        # Get lists of unique cities and stations for rules/metadata
        cities = latest_city_data['City'].unique().tolist()
        stations = latest_station_data['StationName'].unique().tolist()

        # RETURNED DATAFRAMES: Full dataframes (latest_city_data, latest_station_data)
        # are kept for UI rendering and proactive monitoring, while the sampled context
        # (city_context, station_context) is used for the LLM API call.
        return city_context, station_context, latest_city_data, latest_station_data, cities, stations

    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Please ensure all files ({', '.join(files.values())}) are in the directory.")
        return None, None, pd.DataFrame(), pd.DataFrame(), [], []
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, pd.DataFrame(), pd.DataFrame(), [], []

# ==============================
# 3. GROQ CHAT FUNCTION
# ==============================
def get_groq_response(city_context, station_context, cities, stations, user_query):
    """
    Sends the user query along with the sampled AQI data context to the Groq API.
    """
    if not GROQ_API_KEY:
        return "⚠️ Error: GROQ_API_KEY not found. Ensure it is set in your .env file and the 'python-dotenv' library is installed."

    try:
        client = Groq(api_key=GROQ_API_KEY)

        # Construct a detailed system prompt to guide the LLM's behavior and ground its response
        system_prompt = f"""
        You are an Air Quality Assistant Chatbot powered by Groq LLM ({MODEL_NAME}).
        Your purpose is to provide accurate, real-time insights about air pollution
        and public health awareness based *strictly* on the datasets provided below.

        --- CITY-LEVEL DATA (Latest Available Snapshot - SAMPLED) ---
        The data below is a small sample, including the highest-risk city and a few others.
        {city_context}

        --- STATION-LEVEL DATA (Latest Available Snapshot - SAMPLED) ---
        The data below is a small sample of monitoring stations.
        {station_context}

        Available Cities (Total {len(cities)}): {', '.join(cities[:5])} ...
        Available Stations (Total {len(stations)}): {', '.join(stations[:5])} ...

        RULES:
        1. Use the data in the tables above to answer AQI-related questions.
        2. If the user asks for a specific city or station *not* explicitly mentioned in the small sample data tables, you may use the full city/station lists provided as reference, but **you must state that the specific AQI data is not visible in the current sampled context.**
        3. For general AQI awareness (e.g., "What is a good AQI?"), use factual information about AQI categories and health advice.
        4. Keep your tone informative, citizen-friendly, and your responses concise.
        """

        chat_completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        # The error is now more likely to be a rate limit or another API issue,
        # but the context window error should be fixed.
        return f"An API error occurred: {e}. The data context size has been reduced. If this persists, check your Groq API tier limits."

# ==============================
# 4. STREAMLIT APP LAYOUT
# ==============================
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="🌫️ LLM Air Quality Chatbot", page_icon="💨", layout="wide")
    
    st.title(f"🌫️ Leveraging LLMs for Real-Time Air Pollution Monitoring")
    st.caption(f"Powered by **Groq {MODEL_NAME}** and the AQI360 dataset (context size optimized)")

    # Load datasets
    result = load_aqi_datasets(DATA_FILES)
    if result is None or result[2].empty:
        return
        
    city_context, station_context, latest_city_df, latest_station_df, cities, stations = result


    # --- PROACTIVE HEALTH ADVISORY SECTION ---
    st.markdown("---")
    st.subheader("🚨 Proactive Monitoring Alert")
    
    # Identify the city with the worst air quality (highest AQI)
    worst_city_data = latest_city_df.loc[latest_city_df['AQI'].idxmax()]
    
    city_name = worst_city_data['City']
    aqi_val = worst_city_data['AQI']
    aqi_bucket = worst_city_data['AQI_Bucket']
    date_val = worst_city_data['Date'].strftime('%Y-%m-%d')

    alert_message = ""
    
    if aqi_val >= 401:
        alert_message = f"**SEVERE WARNING: Hazardous Air Quality in {city_name}** (AQI {aqi_val}, Category: {aqi_bucket} on {date_val}). **Health Advice:** Avoid all outdoor physical activity. Keep windows and doors closed. Use air purifiers. Everyone should wear a proper respirator (N95/P100) outdoors."
        st.error(alert_message)
    elif aqi_val >= 301:
        alert_message = f"**CRITICAL ALERT: Very Poor Air Quality in {city_name}** (AQI {aqi_val}, Category: {aqi_bucket} on {date_val}). **Health Advice:** Everyone should avoid prolonged or heavy exertion outdoors. People with heart or lung disease, older adults, and children should avoid all outdoor activity."
        st.error(alert_message)
    elif aqi_val >= 201:
        alert_message = f"**HIGH ALERT: Poor Air Quality in {city_name}** (AQI {aqi_val}, Category: {aqi_bucket} on {date_val}). **Health Advice:** Sensitive groups should avoid outdoor activity. General public should limit prolonged outdoor exertion."
        st.warning(alert_message)
    elif aqi_val >= 101:
        alert_message = f"**NOTICE: Moderate Air Quality in {city_name}** (AQI {aqi_val}, Category: {aqi_bucket} on {date_val}). **Health Advice:** Unusually sensitive people should consider limiting prolonged outdoor exertion."
        st.info(alert_message)
    else:
        alert_message = f"**Good Air Quality Check:** The worst recorded AQI is currently {aqi_val} in {city_name}. Air quality is generally acceptable across monitored regions."
        st.success(alert_message)
    
    st.markdown("---")
    # --- END PROACTIVE HEALTH ADVISORY SECTION ---


    # Initialize chat session
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": f"Hello! I have the latest AQI data for **{len(cities)} cities** and **{len(stations)} monitoring stations**. The current highest risk alert is: **{aqi_bucket}** in **{city_name}**. How can I assist you with air quality insights today?"}
        ]

    # --- MAIN CONTENT AREA: CHAT INTERFACE ---
    chat_container = st.container()

    with chat_container:
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Handle user query
        if user_query := st.chat_input("Ask about air quality (e.g., 'What’s the AQI in Delhi?' or 'Which city has the worst air today?')"):
            st.session_state.messages.append({"role": "user", "content": user_query})
            log_chat_message("user", user_query) # Log user message

            with st.chat_message("user"):
                st.markdown(user_query)

            # Get response from Groq
            with st.chat_message("assistant"):
                with st.spinner(f"Analyzing air quality data with Groq ({MODEL_NAME})..."):
                    reply = get_groq_response(city_context, station_context, cities, stations, user_query)
                st.markdown(reply)
                log_chat_message("assistant", reply) # Log assistant response
                
            st.session_state.messages.append({"role": "assistant", "content": reply})

    # --- SIDEBAR FOR DATA VISIBILITY AND BATCH PROCESSING ---
    with st.sidebar:
        st.header("📊 Data Context for LLM")
        st.info("The language model uses the data below as its knowledge base to answer your questions. This data is a token-optimized sample of the full dataset.")
        
        # Display City Data (Using the full data for the UI, but showing the *sampled* context for the LLM)
        st.subheader("🏙️ City Data Snapshot (UI)")
        st.dataframe(latest_city_df[['City', 'Date', 'AQI', 'AQI_Bucket']].head(5), use_container_width=True)
        with st.expander("View LLM Context Table (Sampled)"):
            st.code(city_context, language='markdown')

        # Display Station Data
        st.subheader("📡 Station Data Snapshot (UI)")
        st.dataframe(latest_station_df[['StationName', 'City', 'AQI', 'AQI_Bucket']].head(5), use_container_width=True)
        with st.expander("View LLM Context Table (Sampled)"):
            st.code(station_context, language='markdown')
            
        st.markdown("---")
        
        # --- BATCH ANALYSIS TOOL ---
        st.header("🔬 Batch Analysis Tool")
        st.caption("Process multiple queries sequentially for research logging.")
        
        batch_queries_input = st.text_area(
            "Enter Queries (One per line):", 
            key="batch_queries", 
            height=200,
            value="What is the worst city's AQI?\nCompare PM2.5 and PM10 in the city with the highest AQI.\nWhat health advice is relevant today?"
        )
        
        if st.button("Run Batch Analysis"):
            queries = [q.strip() for q in batch_queries_input.split('\n') if q.strip()]
            if queries:
                st.write(f"**Processing {len(queries)} Queries...**")
                
                # Create a placeholder for results
                results_placeholder = st.empty()
                
                all_batch_results = []
                for i, query in enumerate(queries):
                    st.write(f"**Query {i+1}/{len(queries)}:** {query}")
                    log_chat_message("batch_user", query) # Log batch user message

                    # Get response from Groq
                    with st.spinner(f"Processing Query {i+1}..."):
                        batch_reply = get_groq_response(city_context, station_context, cities, stations, query)
                    
                    st.markdown(f"**Response:** {batch_reply}")
                    log_chat_message("batch_assistant", batch_reply) # Log batch assistant response
                    
                    all_batch_results.append({
                        "query": query,
                        "response": batch_reply
                    })
                
                st.success("Batch Analysis Complete! Results logged to `chat_log.csv`.")
                
            else:
                st.warning("Please enter queries to run the batch analysis.")
                
        st.markdown("---")
        st.caption(f"LLM: {MODEL_NAME}")
        st.caption("Required files: `city_day.csv`, `city_hour.csv`, `station_day.csv`, `station_hour.csv`, `stations.csv`.")
        
# Run the application
if __name__ == "__main__":
    main()
