import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
