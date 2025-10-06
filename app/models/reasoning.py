from pydantic import BaseModel
from typing import List, Optional


class ReasoningStep(BaseModel):
     possible_conditions: List[str]   # e.g. ["Flu", "Bronchitis"]
     recommended_questions: List[str] # e.g. ["How long have you had a fever?", "Any chest tightness?"]
     urgency: str                     # e.g. "low", "medium", "high"
