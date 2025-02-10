import os
import sqlite3
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class SQLQueryAgent:
    def __init__(self, schema: Dict[str, str], database_path: str):
        self.schema = schema
        self.database_path = database_path

    def generate_sql_query(self, user_query: str, table_name: str) -> str:
        """
        Generate SQL query using Groq API with error handling.
        """
        try:
            llm = ChatGroq(
                temperature=0,
                model_name="llama3-70b-8192",
                groq_api_key=os.getenv("GROQ_KEY")
            )

            query_prompt = PromptTemplate.from_template("""
                Schema: {schema}

                User Query: {user_query}

                Generate a valid SQLite query to answer the user's query.
                - Make sure to use the correct spellings according to English Language even if the query has some spelling mistakes.
                - Use '{table_name}' as the table name.
                - Do not add any extra formatting or characters (like backticks, comments, or explanations).
                - Only output the SQLite query as plain text.
            """)

            chain = query_prompt | llm | StrOutputParser()

            raw_response = chain.invoke({
                "schema": str(self.schema),
                "user_query": user_query,
                "table_name": table_name,
            })

            sql_query = raw_response.strip().strip('`').strip()

            if not sql_query:
                raise ValueError("Generated query is empty or invalid.")

            return sql_query

        except ValueError as ve:
            print(f"ValueError: {ve}")
            return "SELECT * FROM sales_table LIMIT 10;"

        except Exception as e:
            print(f"Error generating SQL query: {e}")
            return "SELECT * FROM sales_table LIMIT 10;"

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                columns = [column[0] for column in cursor.description]
                results = [dict(zip(columns, row))
                           for row in cursor.fetchall()]
            return results
        except Exception as e:
            print(f"Query execution error: {e}")
            return []