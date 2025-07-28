"""
Risk Management Package
======================

Système complet de gestion des risques pour le trading algorithmique.
Inclut la surveillance en temps réel, les stop-loss, le contrôle du
drawdown et les circuit breakers.

Composants:
    - RiskMonitor: Surveillance globale des risques
    - PositionSizer: Calcul optimal des tailles de position
    - StopLossManager: Gestion des stop-loss multi-types
    - DrawdownController: Contrôle et limitation du drawdown
    - CorrelationMonitor: Surveillance des corrélations
    - CircuitBreakers: Coupe-circuits automatiques

Usage:
    from risk import RiskMonitor, PositionSizer, StopLossManager
    
    risk_monitor = RiskMonitor(config)
    position_size = await risk_monitor.calculate_position_size(signal)
"""

from risk.risk_monitor import (
    RiskMonitor,
    RiskLevel,
    RiskMetrics,
    RiskAlert,
    RiskLimits,
    RiskState,
    PortfolioRisk
)

from risk.position_sizer import (
    PositionSizer,
    SizingMethod,
    PositionLimits,
    KellyCalculator,
    OptimalF,
    VolatilityScaler,
    RiskParity
)

from risk.stop_loss import (
    StopLossManager,
    StopLossType,
    StopLossOrder,
    StopLossConfig,
    StopLossStatus,
    TrailingMethod,
    StopLossStrategy
)

from risk.drawdown_control import (
    DrawdownController,
    DrawdownLevel,
    DrawdownType,
    DrawdownEvent,
    DrawdownStats,
    RecoveryStatus,
    ActionType,
    DrawdownLimit
)

from risk.correlation_monitor import (
    CorrelationMonitor,
    CorrelationType,
    CorrelationMatrix,
    RegimeType,
    DiversificationMetrics,
    AssetClusterer,
    CorrelationAlert
)

from risk.circuit_breakers import (
    CircuitBreakerManager,
    BreakerType,
    BreakerState,
    TripCondition,
    CircuitBreaker,
    EmergencyAction,
    SystemHealthCheck
)

# Version
__version__ = "1.0.0"

# Exports publics
__all__ = [
    # Risk Monitor
    "RiskMonitor",
    "RiskLevel",
    "RiskMetrics",
    "RiskAlert",
    "RiskLimits",
    "RiskState",
    "PortfolioRisk",
    
    # Position Sizer
    "PositionSizer",
    "SizingMethod",
    "PositionLimits",
    "KellyCalculator",
    "OptimalF",
    "VolatilityScaler",
    "RiskParity",
    
    # Stop Loss
    "StopLossManager",
    "StopLossType",
    "StopLossOrder",
    "StopLossConfig",
    "StopLossStatus",
    "TrailingMethod",
    "StopLossStrategy",
    
    # Drawdown Control
    "DrawdownController",
    "DrawdownLevel",
    "DrawdownType",
    "DrawdownEvent",
    "DrawdownStats",
    "RecoveryStatus",
    "ActionType",
    "DrawdownLimit",
    
    # Correlation Monitor
    "CorrelationMonitor",
    "CorrelationType",
    "CorrelationMatrix",
    "RegimeType",
    "DiversificationMetrics",
    "AssetClusterer",
    "CorrelationAlert",
    
    # Circuit Breakers
    "CircuitBreakerManager",
    "BreakerType",
    "BreakerState",
    "TripCondition",
    "CircuitBreaker",
    "EmergencyAction",
    "SystemHealthCheck"
]

# Configuration par défaut du système de risque
DEFAULT_RISK_CONFIG = {
    "max_portfolio_risk": 0.02,  # 2% VaR quotidien
    "max_position_risk": 0.001,  # 0.1% par position
    "max_correlation": 0.8,      # Corrélation maximale
    "max_drawdown": 0.2,         # 20% drawdown max
    "max_leverage": 3.0,         # Levier maximum
    "confidence_level": 0.99,    # Niveau de confiance VaR
    "risk_free_rate": 0.02      # Taux sans risque annuel
}

# Limites de risque par défaut
DEFAULT_RISK_LIMITS = {
    "position_limits": {
        "max_positions": 50,
        "max_position_size": 0.1,     # 10% du capital
        "max_sector_exposure": 0.3,   # 30% par secteur
        "max_correlated_positions": 5  # Positions fortement corrélées
    },
    "drawdown_limits": {
        "warning": 0.05,      # 5%
        "reduce": 0.10,       # 10%
        "pause": 0.15,        # 15%
        "liquidate": 0.20     # 20%
    },
    "loss_limits": {
        "daily_loss_limit": 0.02,     # 2% par jour
        "weekly_loss_limit": 0.05,    # 5% par semaine
        "monthly_loss_limit": 0.10,   # 10% par mois
        "consecutive_losses": 5       # Nombre max de pertes consécutives
    },
    "exposure_limits": {
        "gross_exposure": 2.0,        # 200% (long + short)
        "net_exposure": 1.0,          # 100% net
        "concentration_limit": 0.25,  # 25% dans un seul actif
        "liquidity_ratio": 0.3       # 30% en actifs liquides minimum
    }
}


class RiskManager:
    """
    Gestionnaire de risque unifié intégrant tous les composants.
    Point d'entrée principal pour la gestion des risques.
    """
    
    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_RISK_CONFIG, **(config or {})}
        self.limits = {**DEFAULT_RISK_LIMITS, **(self.config.get('limits', {}))}
        
        # Initialiser les composants
        self.monitor = RiskMonitor(self.config)
        self.position_sizer = PositionSizer(self.config)
        self.stop_loss_manager = StopLossManager(self.config)
        self.drawdown_controller = DrawdownController(self.config)
        self.correlation_monitor = CorrelationMonitor(self.config)
        self.circuit_breakers = CircuitBreakerManager(self.config)
        
        self._running = False
        
    async def start(self):
        """Démarre tous les composants de risque"""
        await self.monitor.start()
        await self.stop_loss_manager.start()
        await self.drawdown_controller.start()
        await self.correlation_monitor.start()
        await self.circuit_breakers.start()
        self._running = True
        
    async def stop(self):
        """Arrête tous les composants"""
        self._running = False
        await self.monitor.stop()
        await self.stop_loss_manager.stop()
        await self.drawdown_controller.stop()
        await self.correlation_monitor.stop()
        await self.circuit_breakers.stop()
        
    async def check_trade(self, trade_proposal: dict) -> tuple[bool, dict]:
        """
        Vérifie si un trade proposé respecte toutes les règles de risque.
        
        Returns:
            (approved, risk_assessment)
        """
        # Vérifications en cascade
        checks = [
            self.monitor.check_trade_risk,
            self.position_sizer.validate_size,
            self.drawdown_controller.check_drawdown_limits,
            self.correlation_monitor.check_correlation_impact,
            self.circuit_breakers.check_breaker_status
        ]
        
        risk_assessment = {
            "timestamp": datetime.now(timezone.utc),
            "trade": trade_proposal,
            "checks": []
        }
        
        for check in checks:
            result = await check(trade_proposal)
            risk_assessment["checks"].append(result)
            
            if not result.get("approved", False):
                return False, risk_assessment
                
        return True, risk_assessment


# Instance globale optionnelle
_risk_manager = None


def get_risk_manager(config: dict = None) -> RiskManager:
    """Obtient l'instance globale du gestionnaire de risque"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager(config)
    return _risk_manager


# Helpers pour calculs de risque courants
def calculate_var(returns, confidence_level=0.99, method='historical'):
    """Calcule la Value at Risk"""
    import numpy as np
    from scipy import stats
    
    if method == 'historical':
        return np.percentile(returns, (1 - confidence_level) * 100)
    elif method == 'parametric':
        return stats.norm.ppf(1 - confidence_level, np.mean(returns), np.std(returns))
    else:
        raise ValueError(f"Unknown VaR method: {method}")


def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """Calcule le ratio de Sharpe"""
    import numpy as np
    
    excess_returns = returns - risk_free_rate / periods_per_year
    if np.std(returns) == 0:
        return 0
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(returns)


# Import des utilitaires
from datetime import datetime, timezone

# Initialisation au chargement
import logging
logger = logging.getLogger(__name__)
logger.info(f"Risk management package initialized - Version {__version__}")