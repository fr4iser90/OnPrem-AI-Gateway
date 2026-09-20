"""Platform routes — admin/ops for platform administrators."""
from fastapi import APIRouter

from .setup import router as setup_router
from .dashboard import router as dashboard_router
from .users import router as users_router
from .models import router as models_router
from .services import router as services_router
from .settings import router as settings_router
from .usage import router as usage_router

router = APIRouter()
router.include_router(setup_router)
router.include_router(dashboard_router)
router.include_router(users_router)
router.include_router(models_router)
router.include_router(services_router)
router.include_router(settings_router)
router.include_router(usage_router)
