from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import (onnx_t5_custom_router,
                     onnx_t5_pretrained_router)
origins = ["*"]
app = FastAPI()
app.include_router(onnx_t5_custom_router.router)
app.include_router(onnx_t5_pretrained_router.router)
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=False,
                   allow_methods=["POST"],
                   allow_headers=["*"])

