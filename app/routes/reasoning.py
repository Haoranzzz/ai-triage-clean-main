from fastapi import APIRouter

router = APIRouter()

@router.get("/reasoning")
def get_reasoning():
    """
    Simple endpoint to verify reasoning route works.
    For now it just returns a static message.
    """
    return {"message": "Reasoning route is live!"}

