import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from colorama import Fore, Style
from agents.intent import IntentAgent
from agents.ecommerce_assistant import NormalEcommerceAssistantAgent
from agents.memory_agent import ConversationMemoryAgent
from agents.response_formatter import ResponseFormatterAgent
from agents.query_generator import SQLQueryAgent
from agents.infer_schema import SchemaInferenceAgent
from utils.helpers import convert_all_columns_to_snake_case
from tabulate import tabulate

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
