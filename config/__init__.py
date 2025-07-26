"""
Module de Configuration du Robot de Trading Algorithmique IA
============================================================

Ce module centralise toute la configuration du système de trading.
Il charge automatiquement les paramètres selon l'environnement d'exécution
et fournit une interface unique pour accéder à la configuration.

Usage:
    from config import get_config, settings
    
    # Accès direct aux settings
    api_key = settings.exchanges.binance.api_key
    
    # Ou via la fonction helper
    config = get_config()
    db_url = config.database.connection_url

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    TradingConfig, 
    Environment,
    load_config_from_file,
    get_config_path
)

# Singleton pour la configuration globale
_config: Optional[TradingConfig] = None
_config_lock = False


def get_config(reload: bool = False) -> TradingConfig:
    """
    Récupère la configuration singleton du système.
    
    Args:
        reload: Force le rechargement de la configuration
        
    Returns:
        Instance de TradingConfig avec tous les paramètres
        
    Raises:
        RuntimeError: Si la configuration ne peut pas être chargée
    """
    global _config, _config_lock
    
    if _config is not None and not reload:
        return _config
    
    if _config_lock:
        raise RuntimeError("Configuration is being loaded, circular dependency detected")
    
    try:
        _config_lock = True
        
        # Déterminer l'environnement
        env = os.getenv("TRADING_ENV", Environment.DEVELOPMENT.value)
        
        # Charger la configuration selon l'environnement
        config_path = get_config_path(env)
        
        if config_path and config_path.exists():
            _config = load_config_from_file(config_path)
        else:
            # Configuration par défaut depuis les variables d'environnement
            _config = TradingConfig()
        
        # Validation supplémentaire pour la production
        if _config.environment == Environment.PRODUCTION:
            _validate_production_config(_config)
        
        return _config
        
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {str(e)}")
    finally:
        _config_lock = False


def _validate_production_config(config: TradingConfig) -> None:
    """
    Validation stricte pour l'environnement de production.
    
    Args:
        config: Configuration à valider
        
    Raises:
        ValueError: Si la configuration est invalide pour la production
    """
    errors = []
    
    # Vérifier que les modes sandbox sont désactivés
    for exchange in config.exchanges.values():
        if exchange.sandbox_mode:
            errors.append(f"Sandbox mode is enabled for {exchange.name}")
    
    # Vérifier les limites de risque
    if config.risk.max_drawdown_percent > 30:
        errors.append("Max drawdown is too high for production (>30%)")
    
    if config.risk.max_position_size > 0.1:
        errors.append("Max position size is too high for production (>10%)")
    
    # Vérifier la configuration de monitoring
    if not config.monitoring.alerts_enabled:
        warnings.warn("Alerts are disabled in production!")
    
    if errors:
        raise ValueError(f"Production config validation failed: {'; '.join(errors)}")


def update_config(updates: Dict[str, Any]) -> TradingConfig:
    """
    Met à jour la configuration avec de nouvelles valeurs.
    
    Args:
        updates: Dictionnaire des mises à jour
        
    Returns:
        Configuration mise à jour
        
    Note:
        Cette fonction ne doit être utilisée que pour les tests
        ou les ajustements runtime non critiques.
    """
    config = get_config()
    
    # Créer une nouvelle instance avec les updates
    config_dict = config.dict()
    
    # Appliquer les mises à jour de manière récursive
    def deep_update(d: dict, u: dict) -> dict:
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    updated_dict = deep_update(config_dict, updates)
    
    # Recréer la configuration
    global _config
    _config = TradingConfig(**updated_dict)
    
    return _config


# Alias pratiques pour l'import direct
settings = get_config()


# Exports publics du module
__all__ = [
    'get_config',
    'settings',
    'update_config',
    'TradingConfig',
    'Environment',
    'load_config_from_file',
    'get_config_path'
]


# Chargement automatique au démarrage
try:
    # Charger la configuration au démarrage du module
    _ = get_config()
except Exception as e:
    warnings.warn(f"Failed to load config at module init: {str(e)}")