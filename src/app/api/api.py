from app.api.endpoints import auth, callback, cluster, job, photo
from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(job.router, tags=["Clustering Job"])
api_router.include_router(cluster.router, tags=["Clusters"])
api_router.include_router(photo.router, tags=["Photos"])
api_router.include_router(callback.router, prefix="/internal", tags=["Cluster Callback"])