from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine
from routers import appointments, admin
from auth import router as auth_router

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Trauma Team Backend API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for public testing — lock later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router)
app.include_router(appointments.router, prefix="/api/appointments", tags=["Appointments"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])


@app.get("/")
def home():
    return {"status": "Backend online", "service": "TraumaTeam API"}
