import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.auth.earthdata import auth_manager
from app.routers.nasa import router as nasa_router
from app.routers.analysis import router as analysis_router
from app.routers.jobs import router as jobs_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("parallax.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing PARALLAX backend...")
    status = auth_manager.authenticate()
    logger.info(
        "NASA Earthdata Initialization: %s (Status: %s)",
        status["status"],
        "Authenticated" if status["authenticated"] else "Demo/Public Search Active"
    )
    yield
    logger.info("Shutting down PARALLAX backend...")

app = FastAPI(
    title="PARALLAX API — NASA Earthdata Integration",
    version="1.0.0",
    description=(
        "Production backend for NASA Space Apps PARALLAX. "
        "Integrates OPERA L2 RTC-S1 and L3 DSWx-S1 satellite datasets, "
        "executes dual-signature radar backscatter flood/inundated-vegetation detection, "
        "and provides cross-sensor gap intelligence."
    ),
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for data directory (GeoJSON, rasters, cases)
if settings.DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(settings.DATA_DIR)), name="data")

# Register API Routers
app.include_router(nasa_router)
app.include_router(analysis_router)
app.include_router(jobs_router)

@app.get("/")
def root():
    return {
        "service": "PARALLAX NASA Earthdata Service",
        "status": "online",
        "docs_url": "/docs",
        "supported_datasets": ["OPERA_L3_DSWX-S1_V1", "OPERA_L2_RTC-S1_V1"]
    }

@app.get("/health")
def healthcheck():
    return {
        "status": "healthy",
        "earthdata_configured": bool(settings.EARTHDATA_USERNAME and settings.EARTHDATA_PASSWORD),
        "demo_scenes_ready": True
    }
