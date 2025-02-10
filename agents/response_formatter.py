import os
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
                Summarize the query results into a **direct, insightful response** 
                without unnecessary introductions or filler text.

                **Query:** {query}  
                **Results:** {results}  

                Ensure the response is **concise, structured, and informative**.
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
