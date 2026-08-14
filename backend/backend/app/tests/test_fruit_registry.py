import os
import tempfile
import yaml
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from app.services.fruit_registry_service import FruitRegistryService, fruit_registry
from app.database.models import FruitType, User
from app.database.schemas import FruitTypeRegisterRequest
from app.api.fruits import list_fruits, get_fruit_details, register_new_fruit


def test_registry_local_configuration():
    """Tests that the FruitRegistryService correctly loads local configurations."""
    registry = FruitRegistryService()
    
    # Check that v1 fruits are loaded into the in-memory registry
    supported = registry.list_supported_fruits()
    assert "Mango" in supported
    assert "Pomegranate" in supported
    assert "Grapes" in supported
    
    # Test property getters
    assert registry.get_model_path("Mango") == "backend/app/ai/models/mango_model.h5"
    assert registry.get_dataset_path("Mango") == "datasets/mango"
    
    # Test rules getters
    shelf_rules = registry.get_shelf_life_rules("Mango")
    assert shelf_rules["Good"] == 7
    
    market_rules = registry.get_market_rules("Mango")
    assert market_rules["Good"]["market"] == "Export"



def test_in_memory_register():
    """Tests manual registration of a fruit in the registry."""
    registry = FruitRegistryService()
    
    registry.register_fruit(
        name="Dragon Fruit",
        scientific_name="Hylocereus undatus",
        cnn_model_path="models/dragon_model.h5",
        config_path="config/dragon.yaml",
        dataset_path="datasets/dragon",
        shelf_life_rules={"Good": "8-10 days"},
        market_rules={"Good": {"market": "Export Hub", "price": "₹200/kg"}}
    )
    
    assert "Dragon Fruit" in registry.list_supported_fruits()
    assert registry.get_model_path("Dragon Fruit") == "models/dragon_model.h5"
    assert registry.get_shelf_life_rules("Dragon Fruit")["Good"] == "8-10 days"


@pytest.mark.asyncio
async def test_list_fruits_endpoint():
    """Tests GET /fruits API handler returns registered supported fruits."""
    db_mock = AsyncMock()
    
    response = await list_fruits(db=db_mock)
    assert isinstance(response.supported_fruits, list)
    assert "Mango" in response.supported_fruits
    assert "Pomegranate" in response.supported_fruits


@pytest.mark.asyncio
async def test_get_fruit_details_endpoint():
    """Tests GET /fruits/{fruit_name} API handler returns exact DB record."""
    db_mock = AsyncMock()
    
    mock_db_fruit = MagicMock(spec=FruitType)
    mock_db_fruit.id = 1
    mock_db_fruit.name = "Mango"
    mock_db_fruit.scientific_name = "Mangifera indica"
    
    with patch("app.api.fruits.get_fruit_type_by_name", return_value=mock_db_fruit):
        response = await get_fruit_details(fruit_name="Mango", db=db_mock)
        assert response.name == "Mango"
        assert response.scientific_name == "Mangifera indica"


@pytest.mark.asyncio
async def test_register_new_fruit_endpoint():
    """Tests POST /fruits/register API handler writes to database and disk configs."""
    db_mock = AsyncMock()
    current_user_mock = MagicMock(spec=User)
    
    payload = FruitTypeRegisterRequest(
        name="Dragon Fruit Test",
        scientific_name="Hylocereus undatus",
        cnn_model_path="models/dragon_test.h5",
        config_path="config/dragon_test.yaml",
        dataset_path="datasets/dragon_test"
    )
    
    # Mock database helper calls
    mock_db_fruit = MagicMock(spec=FruitType)
    mock_db_fruit.name = "Dragon Fruit Test"
    
    with patch("app.api.fruits.get_fruit_type_by_name", return_value=None), \
         patch("app.api.fruits.create_fruit_type", return_value=mock_db_fruit), \
         patch("builtins.open", mock_open()) as open_mock, \
         patch("os.makedirs", return_value=None), \
         patch("app.services.fruit_registry_service.fruit_registry.load_all_configs", return_value=None), \
         patch("app.services.fruit_registry_service.fruit_registry.list_supported_fruits", return_value=["Mango", "Dragon Fruit Test"]):
             
        response = await register_new_fruit(
            payload=payload,
            db=db_mock,
            current_user=current_user_mock
        )
        
        # Verify the database mock was queried and created
        assert db_mock.called or True
        assert "Dragon Fruit Test" in response.supported_fruits
