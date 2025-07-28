"""
Tests Package
============

Suite de tests complète pour le robot de trading algorithmique.
Inclut tests unitaires, d'intégration et de backtesting.

Structure:
    - unit/: Tests unitaires pour chaque module
    - integration/: Tests d'intégration système
    - backtests/: Tests de backtesting et validation

Usage:
    pytest tests/                    # Tous les tests
    pytest tests/unit/              # Tests unitaires seulement
    pytest tests/integration/       # Tests d'intégration
    pytest -v --cov=.              # Avec coverage
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import warnings

# Configuration des tests
TEST_CONFIG = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "database": "trading_test",
        "user": "test_user",
        "password": "test_pass"
    },
    "redis": {
        "host": "localhost",
        "port": 6379,
        "db": 15  # DB de test
    },
    "exchanges": {
        "binance": {
            "testnet": True,
            "api_key": "test_api_key",
            "api_secret": "test_api_secret"
        }
    },
    "timeouts": {
        "unit": 10,
        "integration": 60,
        "backtest": 300
    }
}

# Fixtures communes
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Configuration pour les tests"""
    return TEST_CONFIG.copy()


@pytest.fixture
async def clean_database():
    """Nettoie la base de données de test"""
    # TODO: Implémenter le nettoyage
    yield
    # Cleanup après test


@pytest.fixture
async def mock_exchange():
    """Mock d'un exchange pour les tests"""
    class MockExchange:
        async def get_ticker(self, symbol):
            return {"symbol": symbol, "price": 50000.0}
            
        async def place_order(self, order):
            return {"order_id": "test_123", "status": "filled"}
            
    return MockExchange()


# Markers personnalisés
def pytest_configure(config):
    """Configure les markers pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "backtest: marks tests as backtest tests"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )


# Helpers pour les tests
class TestDataGenerator:
    """Génère des données de test"""
    
    @staticmethod
    def generate_ohlcv(periods: int = 100) -> List[Dict[str, Any]]:
        """Génère des données OHLCV de test"""
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        data = []
        base_price = 50000
        now = datetime.now()
        
        for i in range(periods):
            # Simulation prix avec random walk
            change = np.random.normal(0, 0.02)
            base_price *= (1 + change)
            
            high = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low = base_price * (1 - abs(np.random.normal(0, 0.01)))
            close = np.random.uniform(low, high)
            
            data.append({
                'timestamp': now - timedelta(minutes=periods-i),
                'open': base_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.uniform(100, 1000)
            })
            
            base_price = close
            
        return data
    
    @staticmethod
    def generate_orderbook(levels: int = 10) -> Dict[str, List]:
        """Génère un carnet d'ordres de test"""
        import numpy as np
        
        mid_price = 50000
        spread = 0.001
        
        bids = []
        asks = []
        
        for i in range(levels):
            bid_price = mid_price * (1 - spread * (i + 1))
            ask_price = mid_price * (1 + spread * (i + 1))
            
            bid_size = np.random.uniform(0.1, 2.0)
            ask_size = np.random.uniform(0.1, 2.0)
            
            bids.append([bid_price, bid_size])
            asks.append([ask_price, ask_size])
            
        return {"bids": bids, "asks": asks}


# Assertions personnalisées
class TradingAssertions:
    """Assertions spécifiques au trading"""
    
    @staticmethod
    def assert_price_in_range(price: float, min_price: float, max_price: float):
        """Vérifie qu'un prix est dans une plage"""
        assert min_price <= price <= max_price, \
            f"Price {price} not in range [{min_price}, {max_price}]"
    
    @staticmethod
    def assert_positive_pnl(pnl: float, tolerance: float = 0):
        """Vérifie un PnL positif avec tolérance"""
        assert pnl >= -tolerance, f"PnL {pnl} is negative beyond tolerance {tolerance}"
    
    @staticmethod
    def assert_valid_sharpe_ratio(sharpe: float):
        """Vérifie un ratio de Sharpe valide"""
        assert -10 <= sharpe <= 10, f"Sharpe ratio {sharpe} seems unrealistic"
        
    @staticmethod
    def assert_order_filled(order_status: str):
        """Vérifie qu'un ordre est exécuté"""
        valid_statuses = ["filled", "partially_filled"]
        assert order_status in valid_statuses, \
            f"Order status '{order_status}' is not filled"


# Décorateurs pour les tests
def async_test(coro):
    """Décorateur pour les tests async"""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper


def skip_if_no_gpu(func):
    """Skip le test si pas de GPU disponible"""
    import torch
    if not torch.cuda.is_available():
        return pytest.mark.skip(reason="No GPU available")(func)
    return func


def with_timeout(seconds: int):
    """Timeout pour les tests"""
    def decorator(func):
        @pytest.mark.timeout(seconds)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Base classes pour les tests
class BaseTestCase:
    """Classe de base pour les tests"""
    
    @classmethod
    def setup_class(cls):
        """Setup pour la classe de tests"""
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
    def setup_method(self):
        """Setup pour chaque méthode"""
        self.test_data = TestDataGenerator()
        self.assertions = TradingAssertions()
        
    def teardown_method(self):
        """Cleanup après chaque test"""
        pass


class AsyncBaseTestCase(BaseTestCase):
    """Classe de base pour les tests async"""
    
    async def async_setup(self):
        """Setup async"""
        pass
        
    async def async_teardown(self):
        """Teardown async"""
        pass


# Exports
__all__ = [
    "TEST_CONFIG",
    "TestDataGenerator",
    "TradingAssertions",
    "BaseTestCase",
    "AsyncBaseTestCase",
    "async_test",
    "skip_if_no_gpu",
    "with_timeout"
]

# Configuration du path pour les imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))