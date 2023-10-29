from pydantic import BaseModel
from typing import List


class QuestionsSchema(BaseModel):
    answer: str
    questions: List[str]


class GeneratedQuestionsSchema(BaseModel):
    context: str
    generated_questions: List[QuestionsSchema]
