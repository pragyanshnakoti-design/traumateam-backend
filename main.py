from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine
from routers import appointments, admin
from auth import router as auth_router

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Trauma Team Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(appointments.router, prefix="/api/appointments")
app.include_router(admin.router, prefix="/api/admin")

@app.get("/")
def home():
    return {"status": "Backend Online", "docs": "/docs"}
