from pydantic import BaseModel, Field

class IntentClassification(BaseModel):
    intent: str = Field(
        description="Categorize the user prompt as either 'query' or 'general'")
    reasoning: str = Field(
        description="Brief explanation of intent classification")
