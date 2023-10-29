from pydantic import BaseModel


class ContentSchema(BaseModel):
    context: str
