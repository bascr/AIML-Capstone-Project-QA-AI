import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from routers.models.questions_schema import QuestionsSchema


class T5QuestionGenerator:

    def __init__(self, model_file_path: str, tokenizer_file_path: str, config: dict):
        self.model_config = T5ForConditionalGeneration.from_pretrained(model_file_path)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_file_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model_config.to(self.device)
        self.config = config

    def generate_question(self, context, answer) -> QuestionsSchema:
        input_ = f"context: {context} answer: {answer}"
        encoding = self.tokenizer.encode_plus(input_,
                                              max_length=self.config["encoder"]["max_length"],
                                              padding="max_length",
                                              return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        self.model.eval()
        beam_outputs = self.model.generate(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           max_length=self.config["generator"]["max_length"],
                                           early_stopping=True,
                                           num_beams=self.config["generator"]["num_beams"],
                                           num_return_sequences=self.config["generator"]["num_return_sequences"])
        questions_lst = []

        for beam_output in beam_outputs:
            question = self.tokenizer.decode(beam_output,
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
            questions_lst.append(question.replace("question: ", ""))

        return QuestionsSchema(answer=answer, questions=questions_lst)



