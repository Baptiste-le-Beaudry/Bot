"""
Unit Tests Sub-package
=====================

Tests unitaires pour tous les modules du système.
Chaque module a son fichier de test correspondant.

Convention de nommage:
    - test_<module_name>.py pour chaque module
    - Test<ClassName> pour chaque classe de test
    - test_<function_name> pour chaque méthode de test
"""

# Import des classes de base depuis le package parent
from tests import BaseTestCase, AsyncBaseTestCase, TradingAssertions

# Helpers spécifiques aux tests unitaires
def mock_market_data():
    """Retourne des données de marché mockées"""
    return {
        "symbol": "BTCUSDT",
        "price": 50000.0,
        "volume": 1000.0,
        "bid": 49999.0,
        "ask": 50001.0,
        "timestamp": "2025-01-01T00:00:00Z"
    }


def mock_order_response():
    """Retourne une réponse d'ordre mockée"""
    return {
        "order_id": "test_order_123",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": 0.1,
        "price": 50000.0,
        "status": "FILLED",
        "executed_quantity": 0.1,
        "executed_price": 50000.0,
        "commission": 0.0001,
        "timestamp": "2025-01-01T00:00:00Z"
    }


# Configuration spécifique pour les tests unitaires
UNIT_TEST_CONFIG = {
    "timeout": 10,
    "use_mocks": True,
    "isolated": True,
    "parallel": True
}

__all__ = [
    "BaseTestCase",
    "AsyncBaseTestCase",
    "TradingAssertions",
    "mock_market_data",
    "mock_order_response",
    "UNIT_TEST_CONFIG"
]