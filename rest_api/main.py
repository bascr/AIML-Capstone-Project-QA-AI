from fastapi import FastAPI
from routers import t5_custom_router, t5_pretrained_router

app = FastAPI()
app.include_router(t5_custom_router.router)
app.include_router(t5_pretrained_router.router)
