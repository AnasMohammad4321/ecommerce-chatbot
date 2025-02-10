import os
from typing import Any
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.intent_classification import IntentClassification

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
