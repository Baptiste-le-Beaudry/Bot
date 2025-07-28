"""
Integration Tests Sub-package
============================

Tests d'intégration pour vérifier l'interaction entre les composants.
Focus sur les scénarios end-to-end et les flux de données complets.

Structure:
    - test_trading_flow.py: Test du flux complet de trading
    - test_data_pipeline.py: Test du pipeline de données
    - test_risk_system.py: Test du système de risque intégré
"""

import asyncio
import pytest
from typing import Dict, Any
from tests import BaseTestCase, AsyncBaseTestCase, TEST_CONFIG

# Configuration pour les tests d'intégration
INTEGRATION_TEST_CONFIG = {
    **TEST_CONFIG,
    "timeout": 60,
    "use_testnet": True,
    "cleanup_after": True,
    "parallel": False  # Les tests d'intégration ne sont pas parallèles
}


class IntegrationTestCase(AsyncBaseTestCase):
    """Classe de base pour les tests d'intégration"""
    
    @classmethod
    def setup_class(cls):
        """Setup des ressources partagées"""
        super().setup_class()
        # Initialiser les connexions de test
        cls.test_db = None
        cls.test_redis = None
        cls.test_exchange = None
    
    async def setup_test_environment(self):
        """Configure l'environnement de test complet"""
        # TODO: Initialiser DB, Redis, Mock Exchange
        pass
    
    async def cleanup_test_environment(self):
        """Nettoie l'environnement de test"""
        # TODO: Cleanup des ressources
        pass


# Fixtures pour les tests d'intégration
@pytest.fixture
async def trading_system():
    """Système de trading complet pour les tests"""
    from core import TradingEngine, PortfolioManager
    from risk import RiskManager
    from data import DataManager
    
    # Configuration de test
    config = INTEGRATION_TEST_CONFIG.copy()
    
    # Initialiser les composants
    engine = TradingEngine(config)
    portfolio = PortfolioManager(config)
    risk = RiskManager(config)
    data = DataManager(config)
    
    # Démarrer les services
    await engine.start()
    await portfolio.start()
    await risk.start()
    await data.start()
    
    yield {
        "engine": engine,
        "portfolio": portfolio,
        "risk": risk,
        "data": data
    }
    
    # Cleanup
    await engine.stop()
    await portfolio.stop()
    await risk.stop()
    await data.stop()


# Helpers pour les tests d'intégration
async def simulate_market_data_stream(symbols: list, duration: int = 10):
    """Simule un stream de données de marché"""
    import random
    import time
    
    start_time = time.time()
    while time.time() - start_time < duration:
        for symbol in symbols:
            yield {
                "symbol": symbol,
                "price": random.uniform(40000, 60000),
                "volume": random.uniform(100, 1000),
                "timestamp": time.time()
            }
        await asyncio.sleep(0.1)


async def wait_for_condition(condition_func, timeout: int = 30, interval: float = 0.5):
    """Attend qu'une condition soit vraie"""
    start_time = asyncio.get_event_loop().time()
    
    while asyncio.get_event_loop().time() - start_time < timeout:
        if await condition_func():
            return True
        await asyncio.sleep(interval)
    
    return False


# Scénarios de test communs
class TradingScenarios:
    """Scénarios de trading pour les tests"""
    
    @staticmethod
    async def simple_buy_sell_scenario(trading_system: Dict[str, Any]):
        """Scénario simple d'achat et vente"""
        engine = trading_system["engine"]
        
        # Créer un signal d'achat
        buy_signal = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.1,
            "signal_strength": 0.8
        }
        
        # Exécuter l'achat
        buy_order = await engine.execute_signal(buy_signal)
        
        # Attendre l'exécution
        await wait_for_condition(
            lambda: buy_order.status == "FILLED",
            timeout=10
        )
        
        # Créer un signal de vente
        sell_signal = {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "quantity": 0.1,
            "signal_strength": 0.7
        }
        
        # Exécuter la vente
        sell_order = await engine.execute_signal(sell_signal)
        
        # Attendre l'exécution
        await wait_for_condition(
            lambda: sell_order.status == "FILLED",
            timeout=10
        )
        
        return buy_order, sell_order
    
    @staticmethod
    async def risk_limit_scenario(trading_system: Dict[str, Any]):
        """Test des limites de risque"""
        engine = trading_system["engine"]
        risk = trading_system["risk"]
        
        # Configurer une limite de risque basse
        await risk.set_max_position_size(0.05)
        
        # Essayer de placer un ordre trop grand
        large_signal = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 1.0,  # Trop grand
            "signal_strength": 0.9
        }
        
        # Devrait être rejeté
        result = await engine.execute_signal(large_signal)
        
        return result


# Exports
__all__ = [
    "INTEGRATION_TEST_CONFIG",
    "IntegrationTestCase",
    "trading_system",
    "simulate_market_data_stream",
    "wait_for_condition",
    "TradingScenarios"
]