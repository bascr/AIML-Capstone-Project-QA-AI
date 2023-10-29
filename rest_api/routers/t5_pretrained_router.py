from fastapi import APIRouter
from ai_models.t5_question_generator import T5QuestionGenerator
from ai_models.key_concept_extractor import KeyConceptExtractor
from utilities.config_reader import ConfigReader
from routers.models.content_schema import ContentSchema
from routers.models.questions_schema import QuestionsSchema, GeneratedQuestionsSchema
from typing import List


config = ConfigReader().get_config()
key_concept_extractor = KeyConceptExtractor(config=config)
pretrained_model = T5QuestionGenerator(model_file_path=config["t5_pretrained_model"]["model_file_path"],
                                       tokenizer_file_path=config["t5_pretrained_model"]["tokenizer_file_path"],
                                       config=config)

router = APIRouter(
    prefix="/t5/pretrained",
    tags=["pretrained"],
    responses={
        404: { "description": "Resource not found."}
    }
)


@router.get("/generate_questions")
async def generate_questions(content: ContentSchema) -> List[GeneratedQuestionsSchema]:
    sentences_lst = key_concept_extractor.get_key_words(content.context)
    response: List[GeneratedQuestionsSchema] = []
    for sentence in sentences_lst:
        questions_lst: List[QuestionsSchema] = []
        for keyword, score in sentence[1]:
            questions_lst.append(pretrained_model.generate_question(context=sentence[0], answer=keyword))
        response.append(GeneratedQuestionsSchema(context=sentence[0], generated_questions=questions_lst))
    return response
