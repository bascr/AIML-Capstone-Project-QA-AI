from fastapi import FastAPI
from routers import (onnx_t5_custom_router,
                     onnx_t5_pretrained_router)

app = FastAPI()
app.include_router(onnx_t5_custom_router.router)
app.include_router(onnx_t5_pretrained_router.router)

