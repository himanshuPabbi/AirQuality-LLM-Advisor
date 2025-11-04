import pandas as pd
import os
import csv
import time
import sys
from groq import Groq
from dotenv import load_dotenv

# ====================================================================
# --- 1. CONFIGURATION AND INITIAL SETUP ---
# ====================================================================

# NOTE: Since this environment path is known from your previous input,
# we are using it here. Adjust if running locally.
QUERIES_FILE = "all_775_queries.txt"
OUTPUT_LOG_FILE = "groq_performance_data.csv"

# Load environment variables for API key
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# WARNING: If running outside the Streamlit app context, you need to
# manually set the API key in your .env file or hardcode it (not recommended).
if not GROQ_API_KEY:
    print("FATAL ERROR: GROQ_API_KEY not found in environment variables.")
    print("Please ensure you have a .env file with GROQ_API_KEY='your_key_here'.")
    sys.exit(1)

# Using the model specified in the last user input
MODEL_NAME = "llama-3.1-8b-instant"
TEMP_FILE_PATH = "/tmp/groq_context_placeholder.csv" # Placeholder for data context

# ====================================================================
# --- 2. CORE LOGIC FUNCTIONS ---
# ====================================================================

def create_mock_context_file(filepath):
    """
    Creates a temporary mock CSV file to satisfy the structure of the
    original Streamlit code's data loading (though we don't use it here).
    For a real test, this ensures the core Groq response logic is reusable.
    """
    # Create a dummy CSV for the system prompt to avoid complexity
    # We will use a simplified system prompt below instead of the complex one
    pass # We will simplify the system prompt construction directly

def get_groq_response_and_time(user_query):
    """
    Sends a query to the Groq API and measures/captures the Total Response Time (TRT).
    
    Returns:
        tuple: (response_content: str, trt_seconds: float)
        
    Note: This is a streamlined version of your Streamlit function,
    focused purely on the API call and latency capture.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)

        # Since we are not running in Streamlit, we must use a minimal system prompt.
        # This simplifies the test while focusing on raw speed.
        system_prompt = f"You are a helpful and very fast assistant powered by Groq ({MODEL_NAME}). Answer the user's air quality question concisely."

        # Start timer for network call + processing (Total Response Time proxy)
        start_time = time.time() 

        chat_completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        
        # Calculate Total Response Time (TRT)
        # We rely on Groq's internal timing for accuracy, which is available in the usage object.
        if chat_completion.usage and hasattr(chat_completion.usage, 'total_time'):
             trt_seconds = chat_completion.usage.total_time
        else:
             # Fallback to local clock time if Groq usage object is missing time
             trt_seconds = time.time() - start_time 

        response_content = chat_completion.choices[0].message.content
        
        return response_content, trt_seconds

    except Exception as e:
        # Log the error and assign a high latency to signify failure
        error_message = f"API_ERROR: {e}"
        print(f"Error for query: {user_query}. {error_message}")
        return error_message, -1.0 # Use -1.0 to easily identify failed requests

# ====================================================================
# --- 3. MAIN EXECUTION ---
# ====================================================================

def main():
    """Reads queries and executes the Groq latency test."""
    try:
        with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"ERROR: Query file '{QUERIES_FILE}' not found. Ensure you ran the extraction script.")
        return

    if not queries:
        print("No queries found in the file. Exiting.")
        return

    print(f"Starting Groq Latency Test for {len(queries)} queries using {MODEL_NAME}...")

    results = []
    
    # Write header to the output log file
    with open(OUTPUT_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Query_ID', 'User_Query', 'Groq_Response_Content', 'Total_Response_Time_Seconds', 'Status'])

    for i, query in enumerate(queries):
        if (i + 1) % 50 == 0:
            print(f"--- Processed {i + 1}/{len(queries)} queries ---")
        
        response, trt = get_groq_response_and_time(query)
        
        status = "SUCCESS" if trt >= 0 else "FAILURE"
        
        # Append results to the log file immediately (good practice for long runs)
        with open(OUTPUT_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([i + 1, query, response, trt, status])

        # Small delay to respect potential rate limits
        time.sleep(0.1)

    # --- FINAL SUMMARY ---
    df_results = pd.read_csv(OUTPUT_LOG_FILE)
    successful_runs = df_results[df_results['Total_Response_Time_Seconds'] >= 0]
    
    if not successful_runs.empty:
        mean_trt = successful_runs['Total_Response_Time_Seconds'].mean()
        median_trt = successful_runs['Total_Response_Time_Seconds'].median()
    else:
        mean_trt = "N/A"
        median_trt = "N/A"

    print("\n==================================================")
    print("        GROQ LATENCY TEST COMPLETE                ")
    print("==================================================")
    print(f"Total Queries Attempted: {len(queries)}")
    print(f"Successful Groq Calls: {len(successful_runs)}")
    print(f"Mean Groq TRT: {mean_trt} seconds")
    print(f"Median Groq TRT: {median_trt} seconds")
    print(f"Results saved to: {OUTPUT_LOG_FILE}")
    print("\nUse this data to compare against your baseline mean of 14.94s.")


if __name__ == "__main__":
    main()
