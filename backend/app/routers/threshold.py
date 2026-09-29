from fastapi import APIRouter

from ..schemas.models import ThresholdOut, ThresholdIn
from ..services.database import db

router = APIRouter()


@router.get("/threshold", response_model=ThresholdOut)
async def get_threshold():
    return await db.get_current_threshold()


@router.put("/threshold", response_model=ThresholdOut)
async def put_threshold(body: ThresholdIn):
    return await db.save_threshold(body.value, body.reason)
