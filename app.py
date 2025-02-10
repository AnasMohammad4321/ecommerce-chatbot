import os
import sys
import pandas as pd
import sqlite3
from typing import List, Dict, Any, Tuple, Union
from dotenv import load_dotenv
from textblob import TextBlob
import re

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

from colorama import Fore, Style, init
from tabulate import tabulate
from pandas.api.types import is_numeric_dtype, is_object_dtype, CategoricalDtype

import logging
logging.basicConfig(level=logging.ERROR, filename="errors.log")
logging.error("Error message", exc_info=True)

init(autoreset=True)


class SchemaInferenceAgent:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.schema: Dict[str, Dict[str, Union[List[Any], Dict[str, Any]]]] = {}

    def infer_schema(self) -> Dict[str, Dict[str, Union[List[Any], Dict[str, Any]]]]:
        """
        Infers the schema by extracting:
        - Column names (converted to snake_case)
        - Unique values for categorical columns (limited to 10 examples)
        - Summary statistics for numerical columns (min, max, mean, std, unique count)
        
        Returns:
            dict: {column_name: {'type': 'categorical' or 'numerical', 'values' (for categorical) or 'stats' (for numerical)}}
        """
        try:
            df = pd.read_csv(self.file_path)

            # Convert column names to snake_case
            df.columns = [convert_to_snake_case(col) for col in df.columns]

            schema_info = {}

            for column in df.columns:
                if df[column].dtype == 'object' or df[column].nunique() < 20:
                    schema_info[column] = {
                        "type": "categorical",
                        "values": df[column].dropna().unique()[:10].tolist() 
                    }
                else:  
                    schema_info[column] = {
                        "type": "numerical",
                        "stats": {
                            "min": float(df[column].min()) if pd.notna(df[column].min()) else None,
                            "max": float(df[column].max()) if pd.notna(df[column].max()) else None,
                            "mean": float(df[column].mean()) if pd.notna(df[column].mean()) else None,
                            "std": float(df[column].std()) if pd.notna(df[column].std()) else None,
                            "unique_count": int(df[column].nunique())
                        }
                    }

            self.schema = schema_info
            return self.schema

        except Exception as e:
            logging.error(f"Error inferring schema: {e}")
            return {}

    def print_schema_summary(self):
        """
        Prints the inferred schema (column names, types, and unique values or statistics).
        """
        print("\nSchema Summary:")
        for column, details in self.schema.items():
            if details["type"] == "categorical":
                print(f"{column} (Categorical): {details['values']}")
            else:
                stats = details["stats"]
                print(f"{column} (Numerical) - Min: {stats['min']}, Max: {stats['max']}, "
                      f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, Unique: {stats['unique_count']}")

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

class ResponseFormatterAgent:
    @staticmethod
    def format_response(query: str, results: List[Dict[str, Any]]) -> str:
        """
        Format query results using Groq API with error handling.
        """
        try:
            llm = ChatGroq(
                temperature=0.3,
                model_name="llama3-70b-8192",
                groq_api_key=os.getenv("GROQ_KEY")
            )

            response_prompt = PromptTemplate.from_template("""
                Convert these query results into a clear, concise response:

                Original Query: {query}
                Results: {results}
                
                Provide a succinct, informative explanation.
            """)

            chain = response_prompt | llm | StrOutputParser()

            response = chain.invoke({
                "query": query,
                "results": str(results)
            })

            if not response:
                raise ValueError("Generated response is empty or invalid.")

            return response.strip()

        except ValueError as ve:
            print(f"ValueError: {ve}")
            return "Unable to format results. Please try again."

        except Exception as e:
            print(f"Error formatting response: {e}")
            return "An error occurred while processing your request. Please contact support."

class IntentClassification(BaseModel):
    intent: str = Field(
        description="Categorize the user prompt as either 'query' or 'general'")
    reasoning: str = Field(
        description="Brief explanation of intent classification")

class IntentAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv("GROQ_KEY")
        )

    def classify_intent(self, user_prompt: str) -> IntentClassification:
        output_parser = PydanticOutputParser(
            pydantic_object=IntentClassification)

        intent_prompt = PromptTemplate(
            template="""Carefully classify the intent of the following user prompt:

                        User Prompt: {prompt}

                        Classification Guidelines:
                        - 'query' intent: Seeks specific information from the sales database (e.g., sales figures, product details)
                        - 'general' intent: Conversational, greeting, or non-database related queries

                        {format_instructions}

                        Provide your classification:""",
            input_variables=["prompt"],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            }
        )

        chain = intent_prompt | self.llm | output_parser

        try:
            result = chain.invoke({"prompt": user_prompt})
            return result
        except Exception as e:
            print(f"Intent classification error: {e}")
            return IntentClassification(
                intent="general",
                reasoning="Failed to classify intent, defaulting to general"
            )

class NormalEcommerceAssistantAgent:
    def __init__(self):
        """
        Initializes the e-commerce assistant agent, tailored for business stakeholders.
        """
        self.llm = ChatGroq(
            temperature=0.2,
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv("GROQ_KEY")
        )

    def generate_response(self, user_prompt: str, context: str) -> str:
        """
        Generates an insightful response tailored for e-commerce stakeholders.
        
        This chatbot is designed to provide:
        - Business analytics insights
        - Sales trends and patterns
        - Operational and logistics data
        - Customer behavior analysis
        - Inventory management suggestions
        - Competitive insights
        
        Args:
            user_prompt (str): The stakeholder's query.
        
        Returns:
            str: The generated response.
        """

        ecommerce_prompt = PromptTemplate.from_template("""
            You are an AI assistant specialized in **e-commerce business intelligence**. 
            Your role is to assist **business stakeholders, analysts, and decision-makers** 
            by providing **data-driven insights, analytics, and strategic advice.** 
            
            Focus areas include:
            - **Sales performance** (revenue trends, top-selling products, seasonal insights)
            - **Customer behavior** (purchase trends, churn risks, demographics)
            - **Inventory management** (stock levels, forecasting demand, supply chain insights)
            - **Marketing effectiveness** (ad performance, conversion rates, ROI)
            - **Competitive analysis** (market trends, pricing strategies, competitor benchmarking)
            - **Operational efficiency** (order fulfillment times, logistics bottlenecks)

            Your responses must be **precise, actionable, and insightful**, avoiding generic replies. 
            
            User Query: {prompt}
            Previous Conversation Context: {context}
            Provide a structured, analytical, and data-backed response.
        """)

        chain = ecommerce_prompt | self.llm | StrOutputParser()

        response = chain.invoke({"prompt": user_prompt, "context": context})
        return response

class ConversationMemoryAgent:
    def __init__(self, max_memory_length: int = 5):
        self.memory: List[Tuple[str, str]] = []
        self.max_memory_length = max_memory_length

    def add_interaction(self, user_prompt: str, system_response: str):
        self.memory.append((user_prompt, system_response))

        if len(self.memory) > self.max_memory_length:
            self.memory = self.memory[-self.max_memory_length:]

    def get_context(self) -> str:
        context = "Conversation History:\n"
        for user, response in self.memory:
            context += f"User: {user}\nAssistant: {response}\n\n"
        return context


def convert_to_snake_case(column_name: Any) -> str:
    """
    Converts a single column name to snake_case. Handles non-string column names gracefully.
    """
    if not isinstance(column_name, str):
        column_name = str(column_name)
    return re.sub(r'[^a-zA-Z0-9]+', '_', column_name).strip('_').lower()


def convert_all_columns_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all column names of a DataFrame to snake_case.
    """
    df.columns = [convert_to_snake_case(col) for col in df.columns]
    return df


def main():
    load_dotenv()

    CSV_PATH = './data/sample_data.csv'
    DB_PATH = 'sales_database.sqlite'

    schema_agent = SchemaInferenceAgent(CSV_PATH)
    schema = schema_agent.infer_schema()

    df_chunks = pd.read_csv(CSV_PATH, chunksize=1000)
    df = pd.concat(chunk for chunk in df_chunks)
    df = convert_all_columns_to_snake_case(df)
    conn = sqlite3.connect(DB_PATH)
    TABLE_NAME = "sales_table"
    df.to_sql(TABLE_NAME, conn, if_exists='replace',
              index=False, method="multi")
    conn.close()

    intent_agent = IntentAgent()
    normal_assistant = NormalEcommerceAssistantAgent()
    memory_agent = ConversationMemoryAgent()
    # query_processor = QueryProcessor()
    print(f"{Fore.GREEN}E-commerce Chatbot:{Style.RESET_ALL} Hello! How can I assist you today?")

    while True:
        user_query = input(f"{Fore.CYAN}You: {Style.RESET_ALL}")

        if user_query.lower() == 'exit':
            print(f"{Fore.YELLOW}Goodbye! Have a great day!{Style.RESET_ALL}")
            break
        context = memory_agent.get_context()

        intent_classification = intent_agent.classify_intent(user_query)
        print(f"{Fore.MAGENTA}Intent:{Style.RESET_ALL} {Fore.YELLOW}{intent_classification.intent}{Style.RESET_ALL} "
              f"{Fore.LIGHTBLACK_EX}(Reasoning: {intent_classification.reasoning}){Style.RESET_ALL}")

        if intent_classification.intent.lower() == 'query':
            sql_agent = SQLQueryAgent(schema, DB_PATH)
            sql_query = sql_agent.generate_sql_query(user_query, TABLE_NAME)
            print(f"{Fore.BLUE}Generated SQL Query:{Style.RESET_ALL} {sql_query}")

            query_results = sql_agent.execute_query(sql_query)

            response_agent = ResponseFormatterAgent()
            final_response = response_agent.format_response(
                user_query, query_results)

            print(f"{Fore.GREEN}Query Results:{Style.RESET_ALL}")
            print(tabulate(query_results, headers="keys", tablefmt="fancy_grid"))

            print(f"{Fore.GREEN}Assistant:{Style.RESET_ALL} {final_response}")

            memory_agent.add_interaction(user_query, final_response)
        else:
            general_response = normal_assistant.generate_response(
                user_query, context)

            print(f"{Fore.GREEN}Assistant:{Style.RESET_ALL} {general_response}")

            memory_agent.add_interaction(user_query, general_response)


if __name__ == "__main__":
    main()
