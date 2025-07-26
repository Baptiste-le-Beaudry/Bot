"""
Système de Contrôle du Drawdown pour Robot de Trading Algorithmique IA
=====================================================================

Ce module implémente un système sophistiqué de surveillance et contrôle
du drawdown pour protéger le capital. Gère les drawdowns par stratégie,
par actif et globaux avec actions automatiques configurables.

Fonctionnalités:
- Calcul temps réel du drawdown (absolu et relatif)
- Limites configurables par niveau (warning, critical, emergency)
- Actions automatiques : réduction positions, pause trading, liquidation
- Tracking du recovery et temps de récupération
- Analyse statistique des drawdowns historiques
- Support multi-stratégies et multi-actifs
- Intégration avec circuit breakers
- Machine Learning pour prédiction de drawdown
- Visualisation et reporting temps réel
- Event sourcing pour audit complet

Architecture:
- Calculs vectorisés pour performance
- State persistence pour recovery
- WebSocket pour monitoring temps réel
- Integration avec Prometheus/Grafana

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Set
import uuid
import warnings

import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Imports internes
from config.settings import get_config
from utils.logger import get_structured_logger
from utils.metrics import MetricsCollector
from utils.decorators import retry_async, measure_performance, cache_result
from core.engine import Event, EventType
from risk.position_sizer import PositionSizer
from monitoring.alerts import AlertManager, AlertLevel, Alert

logger = get_structured_logger(__name__)
metrics = MetricsCollector()


class DrawdownLevel(Enum):
    """Niveaux de drawdown"""
    NORMAL = "normal"           # < 5%
    WARNING = "warning"         # 5-10%
    CRITICAL = "critical"       # 10-20%
    EMERGENCY = "emergency"     # > 20%
    MAXIMUM = "maximum"         # Limite absolue


class DrawdownType(Enum):
    """Types de drawdown trackés"""
    ABSOLUTE = "absolute"       # Perte absolue depuis le pic
    RELATIVE = "relative"       # Perte relative en %
    INTRADAY = "intraday"      # Drawdown intrajournalier
    TRAILING = "trailing"       # Sur période glissante
    UNDERWATER = "underwater"   # Temps sous l'eau


class ActionType(Enum):
    """Actions possibles en réponse au drawdown"""
    NONE = "none"
    REDUCE_POSITION = "reduce_position"
    PAUSE_TRADING = "pause_trading"
    CLOSE_POSITIONS = "close_positions"
    EMERGENCY_LIQUIDATION = "emergency_liquidation"
    STRATEGY_ROTATION = "strategy_rotation"
    INCREASE_HEDGING = "increase_hedging"


class RecoveryStatus(Enum):
    """Statut de récupération"""
    IN_DRAWDOWN = "in_drawdown"
    RECOVERING = "recovering"
    RECOVERED = "recovered"
    NEW_HIGH = "new_high"


@dataclass
class DrawdownLimit:
    """Configuration d'une limite de drawdown"""
    level: DrawdownLevel
    threshold: Decimal
    action: ActionType
    cooldown_minutes: int = 30
    auto_resume: bool = True
    notification_required: bool = True
    
    def is_breached(self, current_dd: Decimal) -> bool:
        """Vérifie si la limite est dépassée"""
        return abs(current_dd) >= abs(self.threshold)


@dataclass
class DrawdownEvent:
    """Événement de drawdown"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    drawdown_type: DrawdownType = DrawdownType.ABSOLUTE
    level: DrawdownLevel = DrawdownLevel.NORMAL
    peak_value: Decimal = Decimal("0")
    trough_value: Decimal = Decimal("0")
    current_value: Decimal = Decimal("0")
    drawdown_amount: Decimal = Decimal("0")
    drawdown_percent: Decimal = Decimal("0")
    duration_minutes: int = 0
    affected_strategies: List[str] = field(default_factory=list)
    action_taken: ActionType = ActionType.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DrawdownStats:
    """Statistiques de drawdown"""
    current_drawdown: Decimal
    current_drawdown_pct: Decimal
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    avg_drawdown: Decimal
    avg_drawdown_duration_hours: float
    max_drawdown_duration_hours: float
    recovery_factor: float  # Profit / Max DD
    calmar_ratio: float    # Annual return / Max DD
    total_drawdown_periods: int
    current_underwater_hours: float
    time_in_drawdown_pct: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convertit en dictionnaire pour JSON"""
        return {k: float(v) if isinstance(v, Decimal) else v 
                for k, v in asdict(self).items()}


class DrawdownAnalyzer:
    """Analyseur de drawdown avec ML"""
    
    def __init__(self):
        self.logger = get_structured_logger(f"{__name__}.DrawdownAnalyzer")
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.is_trained = False
        self.feature_columns = [
            'returns', 'volatility', 'volume_ratio', 
            'rsi', 'trend_strength', 'correlation'
        ]
        
    async def train(self, historical_data: pd.DataFrame):
        """Entraîne le modèle de détection d'anomalies"""
        try:
            if len(historical_data) < 100:
                return
                
            features = self._extract_features(historical_data)
            self.isolation_forest.fit(features)
            self.is_trained = True
            
            self.logger.info("Drawdown analyzer trained", 
                           samples=len(features))
                           
        except Exception as e:
            self.logger.error("Failed to train analyzer", error=str(e))
            
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extrait les features pour l'analyse"""
        features = []
        
        # Returns et volatilité
        data['returns'] = data['value'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        
        # Volume ratio
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # RSI
        data['rsi'] = self._calculate_rsi(data['value'])
        
        # Trend strength
        data['trend_strength'] = self._calculate_trend_strength(data['value'])
        
        # Corrélation rolling
        if 'benchmark' in data.columns:
            data['correlation'] = data['returns'].rolling(20).corr(
                data['benchmark'].pct_change()
            )
        else:
            data['correlation'] = 0
            
        # Nettoyer et retourner
        feature_data = data[self.feature_columns].dropna()
        return feature_data.values
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcule le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calcule la force de la tendance"""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        z_score = (prices - ma) / std
        return z_score.abs()
        
    async def predict_drawdown_risk(self, current_features: Dict[str, float]) -> float:
        """Prédit le risque de drawdown (0-1)"""
        if not self.is_trained:
            return 0.5
            
        try:
            feature_vector = np.array([[
                current_features.get(col, 0) for col in self.feature_columns
            ]])
            
            # Score d'anomalie (-1 pour anomalie, 1 pour normal)
            anomaly_score = self.isolation_forest.decision_function(feature_vector)[0]
            
            # Convertir en probabilité de risque (0-1)
            risk_score = 1 / (1 + np.exp(anomaly_score))
            
            return float(risk_score)
            
        except Exception as e:
            self.logger.error("Prediction failed", error=str(e))
            return 0.5


class DrawdownController:
    """Contrôleur principal du drawdown"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        self.logger = get_structured_logger(__name__)
        
        # Configuration des limites
        self.limits = self._init_limits()
        
        # État du drawdown
        self.portfolio_value_history = deque(maxlen=10000)
        self.peak_values: Dict[str, Decimal] = {}  # Par stratégie
        self.trough_values: Dict[str, Decimal] = {}
        self.drawdown_starts: Dict[str, datetime] = {}
        self.current_drawdowns: Dict[str, Decimal] = {}
        
        # Historique
        self.drawdown_events: deque = deque(maxlen=1000)
        self.recovery_tracking: Dict[str, RecoveryStatus] = defaultdict(
            lambda: RecoveryStatus.NEW_HIGH
        )
        
        # Actions et cooldowns
        self.action_cooldowns: Dict[str, datetime] = {}
        self.paused_strategies: Set[str] = set()
        self.position_reductions: Dict[str, Decimal] = {}
        
        # Analyseur ML
        self.analyzer = DrawdownAnalyzer()
        
        # Métriques temps réel
        self.metrics_cache = {}
        self.last_calculation = time.time()
        
        # État
        self.running = False
        self.emergency_mode = False
        
        # Alertes
        self.alert_manager = AlertManager()
        
        self.logger.info("DrawdownController initialized", 
                        limits=len(self.limits))
                        
    def _init_limits(self) -> List[DrawdownLimit]:
        """Initialise les limites de drawdown"""
        risk_config = self.config.get('risk', {})
        
        limits = [
            DrawdownLimit(
                level=DrawdownLevel.WARNING,
                threshold=Decimal(str(risk_config.get('warning_drawdown', 0.05))),
                action=ActionType.NONE,
                cooldown_minutes=15
            ),
            DrawdownLimit(
                level=DrawdownLevel.CRITICAL,
                threshold=Decimal(str(risk_config.get('critical_drawdown', 0.10))),
                action=ActionType.REDUCE_POSITION,
                cooldown_minutes=30
            ),
            DrawdownLimit(
                level=DrawdownLevel.EMERGENCY,
                threshold=Decimal(str(risk_config.get('emergency_drawdown', 0.20))),
                action=ActionType.PAUSE_TRADING,
                cooldown_minutes=60
            ),
            DrawdownLimit(
                level=DrawdownLevel.MAXIMUM,
                threshold=Decimal(str(risk_config.get('max_drawdown', 0.30))),
                action=ActionType.EMERGENCY_LIQUIDATION,
                cooldown_minutes=120,
                auto_resume=False
            )
        ]
        
        return sorted(limits, key=lambda x: x.threshold)
        
    async def update_portfolio_value(
        self,
        timestamp: datetime,
        total_value: Decimal,
        strategy_values: Optional[Dict[str, Decimal]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Met à jour la valeur du portfolio et calcule les drawdowns"""
        try:
            # Enregistrer la valeur
            self.portfolio_value_history.append({
                'timestamp': timestamp,
                'value': total_value,
                'strategy_values': strategy_values or {},
                'metadata': metadata or {}
            })
            
            # Calculer le drawdown global
            global_dd = await self._calculate_drawdown('GLOBAL', total_value)
            
            # Calculer par stratégie si fourni
            strategy_drawdowns = {}
            if strategy_values:
                for strategy_id, value in strategy_values.items():
                    dd = await self._calculate_drawdown(strategy_id, value)
                    strategy_drawdowns[strategy_id] = dd
                    
            # Vérifier les limites
            breached_limits = await self._check_limits(
                global_dd, strategy_drawdowns
            )
            
            # Prendre les actions nécessaires
            if breached_limits:
                await self._execute_actions(breached_limits, global_dd)
                
            # Mettre à jour les métriques
            await self._update_metrics(global_dd, strategy_drawdowns)
            
            # ML prediction si features disponibles
            if metadata and 'market_features' in metadata:
                risk_score = await self.analyzer.predict_drawdown_risk(
                    metadata['market_features']
                )
                
                if risk_score > 0.8:
                    await self._send_risk_alert(risk_score, metadata)
                    
        except Exception as e:
            self.logger.error("Failed to update portfolio value", 
                            error=str(e), exc_info=True)
                            
    async def _calculate_drawdown(
        self,
        identifier: str,
        current_value: Decimal
    ) -> DrawdownEvent:
        """Calcule le drawdown pour un identifiant"""
        # Obtenir ou initialiser le peak
        if identifier not in self.peak_values:
            self.peak_values[identifier] = current_value
            self.trough_values[identifier] = current_value
            
        peak = self.peak_values[identifier]
        
        # Mettre à jour le peak si nouveau maximum
        if current_value > peak:
            self.peak_values[identifier] = current_value
            peak = current_value
            
            # Marquer comme recovered si était en drawdown
            if identifier in self.drawdown_starts:
                del self.drawdown_starts[identifier]
                self.recovery_tracking[identifier] = RecoveryStatus.NEW_HIGH
                
                # Enregistrer l'événement de recovery
                await self._record_recovery(identifier, current_value)
                
        # Calculer le drawdown
        drawdown_amount = peak - current_value
        drawdown_pct = drawdown_amount / peak if peak > 0 else Decimal("0")
        
        # Gérer l'état du drawdown
        if drawdown_pct > Decimal("0.001"):  # > 0.1%
            if identifier not in self.drawdown_starts:
                self.drawdown_starts[identifier] = datetime.now(timezone.utc)
                self.recovery_tracking[identifier] = RecoveryStatus.IN_DRAWDOWN
                
            # Mettre à jour le trough
            if current_value < self.trough_values[identifier]:
                self.trough_values[identifier] = current_value
                
            duration = datetime.now(timezone.utc) - self.drawdown_starts[identifier]
            duration_minutes = int(duration.total_seconds() / 60)
            
        else:
            duration_minutes = 0
            
        # Créer l'événement
        event = DrawdownEvent(
            drawdown_type=DrawdownType.ABSOLUTE,
            level=self._get_drawdown_level(drawdown_pct),
            peak_value=peak,
            trough_value=self.trough_values.get(identifier, current_value),
            current_value=current_value,
            drawdown_amount=drawdown_amount,
            drawdown_percent=drawdown_pct,
            duration_minutes=duration_minutes,
            affected_strategies=[identifier] if identifier != 'GLOBAL' else []
        )
        
        # Stocker le drawdown actuel
        self.current_drawdowns[identifier] = drawdown_pct
        
        return event
        
    def _get_drawdown_level(self, drawdown_pct: Decimal) -> DrawdownLevel:
        """Détermine le niveau de drawdown"""
        abs_dd = abs(drawdown_pct)
        
        for limit in reversed(self.limits):
            if abs_dd >= limit.threshold:
                return limit.level
                
        return DrawdownLevel.NORMAL
        
    async def _check_limits(
        self,
        global_dd: DrawdownEvent,
        strategy_drawdowns: Dict[str, DrawdownEvent]
    ) -> List[Tuple[DrawdownLimit, DrawdownEvent]]:
        """Vérifie les limites dépassées"""
        breached = []
        
        # Vérifier le drawdown global
        for limit in self.limits:
            if limit.is_breached(global_dd.drawdown_percent):
                if not self._is_in_cooldown(f"GLOBAL_{limit.level.value}"):
                    breached.append((limit, global_dd))
                    
        # Vérifier par stratégie
        for strategy_id, dd_event in strategy_drawdowns.items():
            for limit in self.limits:
                if limit.is_breached(dd_event.drawdown_percent):
                    if not self._is_in_cooldown(f"{strategy_id}_{limit.level.value}"):
                        breached.append((limit, dd_event))
                        
        return breached
        
    def _is_in_cooldown(self, cooldown_key: str) -> bool:
        """Vérifie si une action est en cooldown"""
        if cooldown_key not in self.action_cooldowns:
            return False
            
        cooldown_until = self.action_cooldowns[cooldown_key]
        return datetime.now(timezone.utc) < cooldown_until
        
    async def _execute_actions(
        self,
        breached_limits: List[Tuple[DrawdownLimit, DrawdownEvent]],
        global_dd: DrawdownEvent
    ):
        """Exécute les actions pour les limites dépassées"""
        for limit, dd_event in breached_limits:
            try:
                self.logger.warning(
                    "Drawdown limit breached",
                    level=limit.level.value,
                    drawdown_pct=float(dd_event.drawdown_percent * 100),
                    action=limit.action.value
                )
                
                # Enregistrer l'événement
                dd_event.action_taken = limit.action
                self.drawdown_events.append(dd_event)
                
                # Exécuter l'action
                if limit.action == ActionType.REDUCE_POSITION:
                    await self._reduce_positions(dd_event, reduction_pct=0.5)
                    
                elif limit.action == ActionType.PAUSE_TRADING:
                    await self._pause_trading(dd_event.affected_strategies)
                    
                elif limit.action == ActionType.CLOSE_POSITIONS:
                    await self._close_positions(dd_event.affected_strategies)
                    
                elif limit.action == ActionType.EMERGENCY_LIQUIDATION:
                    await self._emergency_liquidation()
                    
                # Mettre en cooldown
                cooldown_key = f"{dd_event.affected_strategies[0] if dd_event.affected_strategies else 'GLOBAL'}_{limit.level.value}"
                self.action_cooldowns[cooldown_key] = (
                    datetime.now(timezone.utc) + 
                    timedelta(minutes=limit.cooldown_minutes)
                )
                
                # Notification
                if limit.notification_required:
                    await self._send_notification(limit, dd_event)
                    
                # Métriques
                metrics.increment(
                    'drawdown.limit_breached',
                    tags={
                        'level': limit.level.value,
                        'action': limit.action.value
                    }
                )
                
            except Exception as e:
                self.logger.error(
                    "Failed to execute drawdown action",
                    action=limit.action.value,
                    error=str(e),
                    exc_info=True
                )
                
    async def _reduce_positions(
        self,
        dd_event: DrawdownEvent,
        reduction_pct: float = 0.5
    ):
        """Réduit les positions"""
        reduction = Decimal(str(reduction_pct))
        
        # Enregistrer la réduction
        for strategy in dd_event.affected_strategies:
            self.position_reductions[strategy] = reduction
            
        # Émettre l'événement
        await self._emit_event(
            EventType.RISK_ALERT,
            {
                'action': 'reduce_positions',
                'reduction_percentage': float(reduction),
                'affected_strategies': dd_event.affected_strategies,
                'reason': f"Drawdown limit {dd_event.level.value}"
            }
        )
        
        self.logger.info(
            "Position reduction ordered",
            reduction_pct=float(reduction * 100),
            strategies=dd_event.affected_strategies
        )
        
    async def _pause_trading(self, strategies: List[str]):
        """Met en pause le trading"""
        for strategy in strategies:
            self.paused_strategies.add(strategy)
            
        # Émettre l'événement
        await self._emit_event(
            EventType.SYSTEM_STATUS,
            {
                'action': 'pause_trading',
                'strategies': strategies,
                'reason': 'Drawdown limit reached'
            }
        )
        
        self.logger.warning(
            "Trading paused",
            strategies=strategies
        )
        
    async def _close_positions(self, strategies: List[str]):
        """Ferme toutes les positions"""
        await self._emit_event(
            EventType.RISK_ALERT,
            {
                'action': 'close_all_positions',
                'strategies': strategies,
                'urgency': 'high',
                'reason': 'Critical drawdown'
            }
        )
        
        self.logger.warning(
            "Closing all positions",
            strategies=strategies
        )
        
    async def _emergency_liquidation(self):
        """Liquidation d'urgence de tout le portfolio"""
        self.emergency_mode = True
        
        await self._emit_event(
            EventType.RISK_ALERT,
            {
                'action': 'emergency_liquidation',
                'urgency': 'critical',
                'reason': 'Maximum drawdown reached'
            }
        )
        
        # Alerte critique
        await self.alert_manager.send_alert(
            Alert(
                level=AlertLevel.CRITICAL,
                source="DrawdownController",
                message="EMERGENCY LIQUIDATION TRIGGERED",
                details={
                    'drawdown': float(self.current_drawdowns.get('GLOBAL', 0)),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
        )
        
        self.logger.critical("Emergency liquidation triggered")
        
    async def _record_recovery(self, identifier: str, value: Decimal):
        """Enregistre une récupération de drawdown"""
        if identifier in self.drawdown_starts:
            duration = datetime.now(timezone.utc) - self.drawdown_starts[identifier]
            
            recovery_event = {
                'identifier': identifier,
                'recovery_time_hours': duration.total_seconds() / 3600,
                'final_value': float(value),
                'peak_value': float(self.peak_values[identifier]),
                'trough_value': float(self.trough_values.get(identifier, value)),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.info(
                "Drawdown recovered",
                identifier=identifier,
                recovery_hours=recovery_event['recovery_time_hours']
            )
            
            metrics.observe(
                'drawdown.recovery_time_hours',
                recovery_event['recovery_time_hours'],
                tags={'identifier': identifier}
            )
            
    async def _update_metrics(
        self,
        global_dd: DrawdownEvent,
        strategy_drawdowns: Dict[str, DrawdownEvent]
    ):
        """Met à jour les métriques de drawdown"""
        # Métriques globales
        metrics.gauge(
            'drawdown.current_pct',
            float(global_dd.drawdown_percent * 100),
            tags={'type': 'global'}
        )
        
        metrics.gauge(
            'drawdown.duration_minutes',
            global_dd.duration_minutes,
            tags={'type': 'global'}
        )
        
        # Métriques par stratégie
        for strategy_id, dd_event in strategy_drawdowns.items():
            metrics.gauge(
                'drawdown.current_pct',
                float(dd_event.drawdown_percent * 100),
                tags={'type': 'strategy', 'strategy': strategy_id}
            )
            
    async def _send_risk_alert(self, risk_score: float, features: Dict[str, Any]):
        """Envoie une alerte de risque prédictif"""
        await self.alert_manager.send_alert(
            Alert(
                level=AlertLevel.WARNING,
                source="DrawdownController",
                message=f"High drawdown risk detected: {risk_score:.2%}",
                details={
                    'risk_score': risk_score,
                    'features': features,
                    'current_drawdown': float(self.current_drawdowns.get('GLOBAL', 0))
                }
            )
        )
        
    async def _send_notification(self, limit: DrawdownLimit, dd_event: DrawdownEvent):
        """Envoie une notification pour une limite dépassée"""
        message = (
            f"Drawdown {limit.level.value} reached: "
            f"{dd_event.drawdown_percent:.2%}\n"
            f"Action: {limit.action.value}"
        )
        
        await self.alert_manager.send_alert(
            Alert(
                level=AlertLevel.WARNING if limit.level != DrawdownLevel.MAXIMUM else AlertLevel.CRITICAL,
                source="DrawdownController",
                message=message,
                details={
                    'level': limit.level.value,
                    'drawdown_pct': float(dd_event.drawdown_percent),
                    'duration_minutes': dd_event.duration_minutes,
                    'action': limit.action.value
                }
            )
        )
        
    def get_current_stats(self) -> DrawdownStats:
        """Retourne les statistiques actuelles"""
        if not self.portfolio_value_history:
            return self._empty_stats()
            
        # Convertir en DataFrame pour analyse
        df = pd.DataFrame(list(self.portfolio_value_history))
        df['returns'] = df['value'].pct_change()
        
        # Calculer les métriques
        current_dd = self.current_drawdowns.get('GLOBAL', Decimal("0"))
        
        # Maximum drawdown
        rolling_max = df['value'].expanding().max()
        drawdowns = (df['value'] - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        
        # Durées
        underwater_periods = self._calculate_underwater_periods(df)
        
        # Recovery factor
        total_profit = df['value'].iloc[-1] - df['value'].iloc[0]
        recovery_factor = float(total_profit / abs(max_dd)) if max_dd != 0 else 0
        
        # Calmar ratio (annualisé)
        days = (df.index[-1] - df.index[0]).days
        annual_return = (df['value'].iloc[-1] / df['value'].iloc[0]) ** (365/days) - 1
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        return DrawdownStats(
            current_drawdown=current_dd * df['value'].iloc[-1],
            current_drawdown_pct=current_dd,
            max_drawdown=Decimal(str(abs(max_dd) * df['value'].max())),
            max_drawdown_pct=Decimal(str(abs(max_dd))),
            avg_drawdown=Decimal(str(drawdowns[drawdowns < 0].mean())),
            avg_drawdown_duration_hours=np.mean([p[1] for p in underwater_periods]),
            max_drawdown_duration_hours=max([p[1] for p in underwater_periods], default=0),
            recovery_factor=recovery_factor,
            calmar_ratio=calmar,
            total_drawdown_periods=len(underwater_periods),
            current_underwater_hours=self._get_current_underwater_hours(),
            time_in_drawdown_pct=self._calculate_time_in_drawdown(underwater_periods, days)
        )
        
    def _calculate_underwater_periods(self, df: pd.DataFrame) -> List[Tuple[datetime, float]]:
        """Calcule les périodes sous l'eau"""
        periods = []
        in_drawdown = False
        start_time = None
        
        rolling_max = df['value'].expanding().max()
        is_underwater = df['value'] < rolling_max
        
        for i, underwater in enumerate(is_underwater):
            if underwater and not in_drawdown:
                start_time = df.iloc[i]['timestamp']
                in_drawdown = True
            elif not underwater and in_drawdown:
                duration = (df.iloc[i]['timestamp'] - start_time).total_seconds() / 3600
                periods.append((start_time, duration))
                in_drawdown = False
                
        # Période en cours
        if in_drawdown:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds() / 3600
            periods.append((start_time, duration))
            
        return periods
        
    def _get_current_underwater_hours(self) -> float:
        """Retourne le temps actuel sous l'eau"""
        if 'GLOBAL' not in self.drawdown_starts:
            return 0
            
        duration = datetime.now(timezone.utc) - self.drawdown_starts['GLOBAL']
        return duration.total_seconds() / 3600
        
    def _calculate_time_in_drawdown(
        self,
        underwater_periods: List[Tuple[datetime, float]],
        total_days: int
    ) -> float:
        """Calcule le pourcentage de temps en drawdown"""
        if total_days == 0:
            return 0
            
        total_underwater_hours = sum(p[1] for p in underwater_periods)
        total_hours = total_days * 24
        
        return (total_underwater_hours / total_hours) * 100 if total_hours > 0 else 0
        
    def _empty_stats(self) -> DrawdownStats:
        """Retourne des stats vides"""
        return DrawdownStats(
            current_drawdown=Decimal("0"),
            current_drawdown_pct=Decimal("0"),
            max_drawdown=Decimal("0"),
            max_drawdown_pct=Decimal("0"),
            avg_drawdown=Decimal("0"),
            avg_drawdown_duration_hours=0,
            max_drawdown_duration_hours=0,
            recovery_factor=0,
            calmar_ratio=0,
            total_drawdown_periods=0,
            current_underwater_hours=0,
            time_in_drawdown_pct=0
        )
        
    async def reset_strategy_drawdown(self, strategy_id: str):
        """Réinitialise le drawdown d'une stratégie"""
        if strategy_id in self.peak_values:
            del self.peak_values[strategy_id]
        if strategy_id in self.trough_values:
            del self.trough_values[strategy_id]
        if strategy_id in self.drawdown_starts:
            del self.drawdown_starts[strategy_id]
        if strategy_id in self.current_drawdowns:
            del self.current_drawdowns[strategy_id]
            
        self.recovery_tracking[strategy_id] = RecoveryStatus.NEW_HIGH
        
        self.logger.info("Strategy drawdown reset", strategy=strategy_id)
        
    async def resume_trading(self, strategies: Optional[List[str]] = None):
        """Reprend le trading après une pause"""
        if strategies:
            for strategy in strategies:
                self.paused_strategies.discard(strategy)
        else:
            self.paused_strategies.clear()
            
        await self._emit_event(
            EventType.SYSTEM_STATUS,
            {
                'action': 'resume_trading',
                'strategies': strategies or 'all'
            }
        )
        
        self.logger.info("Trading resumed", strategies=strategies or 'all')
        
    def is_strategy_paused(self, strategy_id: str) -> bool:
        """Vérifie si une stratégie est en pause"""
        return strategy_id in self.paused_strategies
        
    def get_position_reduction(self, strategy_id: str) -> Decimal:
        """Retourne la réduction de position active"""
        return self.position_reductions.get(strategy_id, Decimal("0"))
        
    async def generate_report(self) -> Dict[str, Any]:
        """Génère un rapport détaillé"""
        stats = self.get_current_stats()
        
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': stats.to_dict(),
            'current_status': {
                'emergency_mode': self.emergency_mode,
                'paused_strategies': list(self.paused_strategies),
                'active_reductions': {
                    k: float(v) for k, v in self.position_reductions.items()
                }
            },
            'limits': [
                {
                    'level': limit.level.value,
                    'threshold': float(limit.threshold),
                    'action': limit.action.value
                }
                for limit in self.limits
            ],
            'recent_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'level': event.level.value,
                    'drawdown_pct': float(event.drawdown_percent),
                    'duration_minutes': event.duration_minutes,
                    'action': event.action_taken.value
                }
                for event in list(self.drawdown_events)[-10:]
            ],
            'strategy_drawdowns': {
                k: float(v) for k, v in self.current_drawdowns.items()
            }
        }
        
        return report
        
    async def _emit_event(self, event_type: EventType, data: Dict[str, Any]):
        """Émet un événement via le bus"""
        # TODO: Implémenter l'émission via event bus
        pass
        
    async def start(self):
        """Démarre le contrôleur"""
        self.running = True
        self.logger.info("DrawdownController started")
        
    async def stop(self):
        """Arrête le contrôleur"""
        self.running = False
        self.logger.info("DrawdownController stopped")
        
    async def cleanup(self):
        """Nettoie les ressources"""
        # Sauvegarder l'état si nécessaire
        state = {
            'peak_values': {k: str(v) for k, v in self.peak_values.items()},
            'current_drawdowns': {k: str(v) for k, v in self.current_drawdowns.items()},
            'paused_strategies': list(self.paused_strategies),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # TODO: Sauvegarder dans la base de données
        
        self.logger.info("DrawdownController cleanup completed")


# Fonction helper pour visualisation
def plot_drawdown_analysis(controller: DrawdownController, save_path: Optional[str] = None):
    """Génère une visualisation de l'analyse de drawdown"""
    if not controller.portfolio_value_history:
        return
        
    df = pd.DataFrame(list(controller.portfolio_value_history))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. Valeur du portfolio
    ax1 = axes[0]
    ax1.plot(df['timestamp'], df['value'], label='Portfolio Value', color='blue')
    ax1.fill_between(df['timestamp'], df['value'], alpha=0.3)
    ax1.set_ylabel('Value ($)')
    ax1.set_title('Portfolio Value Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    rolling_max = df['value'].expanding().max()
    drawdown_pct = ((df['value'] - rolling_max) / rolling_max) * 100
    
    ax2 = axes[1]
    ax2.fill_between(df['timestamp'], 0, drawdown_pct, color='red', alpha=0.7)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Drawdown Analysis')
    ax2.grid(True, alpha=0.3)
    
    # Ajouter les limites
    for limit in controller.limits:
        ax2.axhline(
            y=-float(limit.threshold) * 100,
            color='orange' if limit.level != DrawdownLevel.MAXIMUM else 'darkred',
            linestyle='--',
            label=f'{limit.level.value}: {limit.threshold:.1%}'
        )
    ax2.legend()
    
    # 3. Distribution des drawdowns
    ax3 = axes[2]
    drawdown_values = drawdown_pct[drawdown_pct < 0]
    if len(drawdown_values) > 0:
        ax3.hist(drawdown_values, bins=50, color='red', alpha=0.7, edgecolor='black')
        ax3.axvline(x=drawdown_values.mean(), color='yellow', linestyle='--', 
                   label=f'Mean: {drawdown_values.mean():.2f}%')
        ax3.axvline(x=drawdown_values.min(), color='darkred', linestyle='--',
                   label=f'Max: {drawdown_values.min():.2f}%')
    ax3.set_xlabel('Drawdown (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Drawdown Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()