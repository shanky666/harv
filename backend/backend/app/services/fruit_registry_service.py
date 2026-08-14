"""
Fruit Registry Service
Dynamically manages supported fruits, their model paths, datasets, and business rules.
"""
import os
import yaml
from typing import Dict, List, Any, Optional
from loguru import logger

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "fruits")

class FruitRegistryService:
    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}
        self.load_all_configs()

    def load_all_configs(self):
        """
        Scans the config directory for YAML files and registers them.
        """
        self._registry.clear()
        if not os.path.exists(CONFIG_DIR):
            logger.warning(f"Fruit config directory does not exist: {CONFIG_DIR}")
            os.makedirs(CONFIG_DIR, exist_ok=True)
            return

        for filename in os.listdir(CONFIG_DIR):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                filepath = os.path.join(CONFIG_DIR, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        if data and "name" in data:
                            self.register_fruit(
                                name=data["name"],
                                scientific_name=data.get("scientific_name", "Unknown"),
                                cnn_model_path=data.get("cnn_model_path", ""),
                                config_path=filepath,
                                dataset_path=data.get("dataset_path", ""),
                                shelf_life_rules=data.get("shelf_life_rules", {}),
                                market_rules=data.get("market_rules", {})
                            )
                except Exception as e:
                    logger.error(f"Failed to load fruit config from {filepath}: {e}")

        logger.info(f"FruitRegistry: Loaded {len(self._registry)} supported fruits: {list(self._registry.keys())}")

    def register_fruit(self, name: str, scientific_name: str, cnn_model_path: str,
                       config_path: str, dataset_path: str, shelf_life_rules: Dict[str, str],
                       market_rules: Dict[str, Any]):
        """
        Registers a fruit in the in-memory registry.
        """
        key = name.lower().strip()
        self._registry[key] = {
            "name": name,
            "scientific_name": scientific_name,
            "cnn_model_path": cnn_model_path,
            "config_path": config_path,
            "dataset_path": dataset_path,
            "shelf_life_rules": shelf_life_rules,
            "market_rules": market_rules
        }
        logger.debug(f"Registered fruit type: {name} ({scientific_name})")

    def get_fruit(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves registered fruit configurations.
        """
        return self._registry.get(name.lower().strip())

    def list_supported_fruits(self) -> List[str]:
        """
        Lists proper-cased names of all registered fruits.
        """
        return [fruit["name"] for fruit in self._registry.values()]

    def get_model_path(self, name: str) -> Optional[str]:
        """
        Returns the CNN model path configured for the fruit.
        """
        fruit = self.get_fruit(name)
        return fruit["cnn_model_path"] if fruit else None

    def get_resolved_model_path(self, name: str) -> Optional[str]:
        """
        Returns the resolved model path for the fruit.
        Resolves via the central Model Registry if active, otherwise falls back to config.
        """
        try:
            from ai_models.model_registry.model_loader import resolve_active_model_path
            key = f"{name.lower().strip()}_model"
            reg_path = resolve_active_model_path(key)
            if reg_path and os.path.exists(reg_path):
                return reg_path
        except Exception:
            pass
            
        return self.get_model_path(name)

    def get_dataset_path(self, name: str) -> Optional[str]:
        """
        Returns the dataset path configured for the fruit.
        """
        fruit = self.get_fruit(name)
        return fruit["dataset_path"] if fruit else None

    def get_shelf_life_rules(self, name: str) -> Optional[Dict[str, str]]:
        """
        Returns the shelf life days mapping rules configured for the fruit.
        """
        fruit = self.get_fruit(name)
        return fruit["shelf_life_rules"] if fruit else None

    def get_market_rules(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Returns the market recommendation rules configured for the fruit.
        """
        fruit = self.get_fruit(name)
        return fruit["market_rules"] if fruit else None


fruit_registry = FruitRegistryService()
