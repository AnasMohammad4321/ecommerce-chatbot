import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from agents.intent import IntentAgent
from agents.ecommerce_assistant import NormalEcommerceAssistantAgent
from agents.memory_agent import ConversationMemoryAgent
from agents.response_formatter import ResponseFormatterAgent
from agents.query_generator import SQLQueryAgent
from agents.infer_schema import SchemaInferenceAgent
from utils.helpers import convert_all_columns_to_snake_case

app = Flask(__name__)

# Load environment variables
load_dotenv()

CSV_PATH = './data/sample_data.csv'
DB_PATH = 'sales_database.sqlite'

# Initialize agents and database
schema_agent = SchemaInferenceAgent(CSV_PATH)
schema = schema_agent.infer_schema()

df_chunks = pd.read_csv(CSV_PATH, chunksize=1000)
df = pd.concat(chunk for chunk in df_chunks)
df = convert_all_columns_to_snake_case(df)
conn = sqlite3.connect(DB_PATH)
TABLE_NAME = "sales_table"
df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False, method="multi")
conn.close()

intent_agent = IntentAgent()
normal_assistant = NormalEcommerceAssistantAgent()
memory_agent = ConversationMemoryAgent()

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    context = memory_agent.get_context()
    intent_classification = intent_agent.classify_intent(user_query)

    if intent_classification.intent.lower() == 'query':
        sql_agent = SQLQueryAgent(schema, DB_PATH)
        sql_query = sql_agent.generate_sql_query(user_query, TABLE_NAME)
        query_results = sql_agent.execute_query(sql_query)

        response_agent = ResponseFormatterAgent()
        final_response = response_agent.format_response(user_query, query_results)

        memory_agent.add_interaction(user_query, final_response)
        return jsonify({"intent": intent_classification.intent, "response": final_response, "results": query_results}), 200
    else:
        general_response = normal_assistant.generate_response(user_query, context)
        memory_agent.add_interaction(user_query, general_response)
        return jsonify({"intent": intent_classification.intent, "response": general_response}), 200

if __name__ == "__main__":
    app.run(debug=True)
