import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agents.intent import IntentAgent
from agents.ecommerce_assistant import NormalEcommerceAssistantAgent
from agents.memory_agent import ConversationMemoryAgent
from agents.response_formatter import ResponseFormatterAgent
from agents.query_generator import SQLQueryAgent
from agents.infer_schema import SchemaInferenceAgent
from utils.helpers import convert_all_columns_to_snake_case

app = FastAPI()

load_dotenv()

CSV_PATH = './data/sample_data.csv'
DB_PATH = 'sales_database.sqlite'

try:
    schema_agent = SchemaInferenceAgent(CSV_PATH)
    schema = schema_agent.infer_schema()

    df_chunks = pd.read_csv(CSV_PATH, chunksize=1000)
    df = pd.concat(chunk for chunk in df_chunks)
    df = convert_all_columns_to_snake_case(df)
    conn = sqlite3.connect(DB_PATH)
    TABLE_NAME = "sales_table"
    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False, method="multi")
    conn.close()
except Exception as e:
    print(f"Error during initialization: {e}")
    exit(1)

intent_agent = IntentAgent()
normal_assistant = NormalEcommerceAssistantAgent()
memory_agent = ConversationMemoryAgent()

class ChatRequest(BaseModel):
    query: str

@app.post('/chat')
async def chat(request: ChatRequest):
    user_query = request.query
    if not user_query:
        raise HTTPException(status_code=400, detail="No query provided")

    context = memory_agent.get_context()
    
    try:
        intent_classification = intent_agent.classify_intent(user_query)

        if intent_classification.intent.lower() == 'query':
            sql_agent = SQLQueryAgent(schema, DB_PATH)
            sql_query = sql_agent.generate_sql_query(user_query, TABLE_NAME)
            query_results = sql_agent.execute_query(sql_query)

            response_agent = ResponseFormatterAgent()
            final_response = response_agent.format_response(user_query, query_results)

            memory_agent.add_interaction(user_query, final_response)
            return JSONResponse(content={"intent": intent_classification.intent, "response": final_response, "results": query_results}, status_code=200)
        else:
            general_response = normal_assistant.generate_response(user_query, context)
            memory_agent.add_interaction(user_query, general_response)
            return JSONResponse(content={"intent": intent_classification.intent, "response": general_response}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
