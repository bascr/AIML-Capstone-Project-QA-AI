from fastT5 import get_onnx_runtime_sessions, OnnxT5
from transformers import T5Tokenizer
from routers.models.questions_schema import QuestionsSchema


class ONNXT5QuestionGenerator:

    def __init__(self, model_path: str, config: dict):
        encoder_path = f"{model_path}/model-encoder-quantized.onnx"
        decoder_path = f"{model_path}/model-decoder-quantized.onnx"
        init_decoder_paths = f"{model_path}/model-init-decoder-quantized.onnx"
        model_file_paths = encoder_path, decoder_path, init_decoder_paths
        model_session = get_onnx_runtime_sessions(model_file_paths)
        self.model = OnnxT5(model_path, model_session)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.config = config

    def generate_question(self, context, answer) -> QuestionsSchema:
        input_ = f"context: {context} answer: {answer}"
        encoding = self.tokenizer.encode_plus(input_,
                                              max_length=self.config["encoder"]["max_length"],
                                              padding="max_length",
                                              return_tensors="pt")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
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
