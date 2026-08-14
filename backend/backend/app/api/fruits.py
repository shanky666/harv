import os
import yaml
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.connection import get_db
from app.database.crud import create_fruit_type, get_fruit_type_by_name, list_fruit_types
from app.database.schemas import FruitTypeOut, FruitTypeRegisterRequest, FruitRegistryListResponse
from app.services.fruit_registry_service import fruit_registry
from app.middleware.authentication import get_current_user
from app.database.models import User

router = APIRouter(prefix="/fruits", tags=["Fruit Registry"])

@router.get("", response_model=FruitRegistryListResponse,
            summary="List all supported fruits in HarvestLenz")
async def list_fruits(db: AsyncSession = Depends(get_db)):
    # Sync registry names
    fruits_list = fruit_registry.list_supported_fruits()
    if not fruits_list:
        # If registry is empty, load from database as fallback
        db_fruits = await list_fruit_types(db)
        fruits_list = [f.name for f in db_fruits]
        
    return FruitRegistryListResponse(supported_fruits=fruits_list)

@router.get("/{fruit_name}", response_model=FruitTypeOut,
            summary="Get scientific details and model configuration for a specific fruit")
async def get_fruit_details(fruit_name: str, db: AsyncSession = Depends(get_db)):
    db_fruit = await get_fruit_type_by_name(db, fruit_name)
    if not db_fruit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fruit '{fruit_name}' is not registered in the database"
        )
    return db_fruit

@router.post("/register", response_model=FruitRegistryListResponse, status_code=status.HTTP_201_CREATED,
             summary="Register a new fruit in the system dynamically (YAML + Database)")
async def register_new_fruit(
    payload: FruitTypeRegisterRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # 1. Check if already registered
    existing = await get_fruit_type_by_name(db, payload.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Fruit '{payload.name}' is already registered in the system"
        )

    # 2. Persist in PostgreSQL database
    db_fruit = await create_fruit_type(
        db=db,
        name=payload.name,
        scientific_name=payload.scientific_name,
        cnn_model_path=payload.cnn_model_path,
        config_path=payload.config_path,
        dataset_path=payload.dataset_path
    )

    # 3. Generate and write the YAML configuration file to persist in config folder
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "fruits")
    os.makedirs(config_dir, exist_ok=True)
    yaml_path = os.path.join(config_dir, f"{payload.name.lower().strip()}.yaml")

    config_data = {
        "name": payload.name,
        "scientific_name": payload.scientific_name,
        "cnn_model_path": payload.cnn_model_path,
        "dataset_path": payload.dataset_path,
        "shelf_life_rules": {
            "Good": "5-7 days",
            "Better": "3-5 days",
            "Medium": "2-3 days",
            "Reject": "0-1 days"
        },
        "market_rules": {
            "Good": {"market": "Export Hub", "price": "₹60-₹80/kg", "alt": []},
            "Better": {"market": "Premium Wholesale", "price": "₹45-₹60/kg", "alt": []},
            "Medium": {"market": "Local Mandi", "price": "₹20-₹35/kg", "alt": []},
            "Reject": {"market": "Processing Plant", "price": "₹5-₹10/kg", "alt": []}
        }
    }

    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Successfully saved to database, but failed to write configuration file: {e}"
        )

    # 4. Reload all configurations in the registry to apply memory updates
    fruit_registry.load_all_configs()

    # 5. Return the updated list of supported fruits
    return FruitRegistryListResponse(supported_fruits=fruit_registry.list_supported_fruits())
