"""
Data Validator - Validation de la Qualité des Données de Trading
================================================================

Ce module implémente un système complet de validation des données de trading
pour garantir la qualité et l'intégrité avant utilisation. Détecte les anomalies,
données manquantes, outliers et incohérences pour éviter les erreurs de trading.

Validations effectuées:
- Intégrité temporelle (gaps, duplications, ordre)
- Cohérence OHLCV (High >= Low, etc.)
- Détection d'outliers statistiques
- Validation des spreads bid/ask
- Continuité des prix (gaps excessifs)
- Volume et liquidité minimale
- Données manquantes ou corrompues
- Anomalies de marché (flash crashes, etc.)

Architecture:
- Validators modulaires par type de données
- Scoring de qualité avec seuils configurables
- Rapports détaillés avec recommandations
- Auto-correction pour problèmes mineurs
- Alertes pour problèmes critiques

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import json
from collections import defaultdict
from scipy import stats
import warnings

# Imports internes
from utils.logger import get_structured_logger
from monitoring.alerts import AlertManager, AlertLevel

warnings.filterwarnings('ignore')


class DataQualityLevel(Enum):
    """Niveaux de qualité des données"""
    EXCELLENT = "excellent"      # >= 95% score
    GOOD = "good"               # >= 85% score
    ACCEPTABLE = "acceptable"   # >= 70% score
    POOR = "poor"              # >= 50% score
    UNUSABLE = "unusable"      # < 50% score


class ValidationSeverity(Enum):
    """Sévérité des problèmes détectés"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataType(Enum):
    """Types de données supportés"""
    TICK = "tick"
    OHLCV = "ohlcv"
    ORDER_BOOK = "order_book"
    TRADES = "trades"


@dataclass
class ValidationIssue:
    """Problème détecté lors de la validation"""
    issue_type: str
    severity: ValidationSeverity
    description: str
    affected_rows: List[int] = field(default_factory=list)
    affected_columns: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    can_fix: bool = False
    fix_suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Rapport complet de validation"""
    data_type: DataType
    total_rows: int
    valid_rows: int
    quality_score: float
    quality_level: DataQualityLevel
    issues: List[ValidationIssue] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    validation_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def invalid_rows(self) -> int:
        return self.total_rows - self.valid_rows
    
    @property
    def validity_percentage(self) -> float:
        return (self.valid_rows / self.total_rows * 100) if self.total_rows > 0 else 0
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Filtre les issues par sévérité"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation"""
        return {
            'data_type': self.data_type.value,
            'total_rows': self.total_rows,
            'valid_rows': self.valid_rows,
            'invalid_rows': self.invalid_rows,
            'validity_percentage': self.validity_percentage,
            'quality_score': self.quality_score,
            'quality_level': self.quality_level.value,
            'n_issues': len(self.issues),
            'critical_issues': len(self.get_issues_by_severity(ValidationSeverity.CRITICAL)),
            'error_issues': len(self.get_issues_by_severity(ValidationSeverity.ERROR)),
            'warning_issues': len(self.get_issues_by_severity(ValidationSeverity.WARNING)),
            'statistics': self.statistics,
            'validation_time_ms': self.validation_time_ms,
            'timestamp': self.timestamp.isoformat()
        }


class BaseValidator:
    """Classe de base pour tous les validators"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_structured_logger(
            f"{self.__class__.__name__}",
            module="data.processors"
        )
        self._setup_default_config()
    
    def _setup_default_config(self) -> None:
        """Configure les paramètres par défaut"""
        pass
    
    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Méthode de validation à implémenter"""
        raise NotImplementedError


class TemporalValidator(BaseValidator):
    """Validateur pour l'intégrité temporelle"""
    
    def _setup_default_config(self) -> None:
        defaults = {
            'max_gap_seconds': 300,           # 5 minutes max entre points
            'min_gap_milliseconds': 100,      # 100ms minimum entre points
            'allow_duplicates': False,
            'require_monotonic': True,
            'business_hours_only': False,
            'timezone': 'UTC'
        }
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Valide l'intégrité temporelle des données"""
        issues = []
        
        if data.empty:
            issues.append(ValidationIssue(
                issue_type="empty_data",
                severity=ValidationSeverity.CRITICAL,
                description="Dataset is empty"
            ))
            return issues
        
        # Vérifier que l'index est datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            issues.append(ValidationIssue(
                issue_type="invalid_index",
                severity=ValidationSeverity.CRITICAL,
                description="Index must be DatetimeIndex",
                can_fix=False
            ))
            return issues
        
        # Vérifier l'ordre chronologique
        if self.config['require_monotonic']:
            if not data.index.is_monotonic_increasing:
                unsorted_positions = np.where(np.diff(data.index.values) < 0)[0]
                issues.append(ValidationIssue(
                    issue_type="non_monotonic_time",
                    severity=ValidationSeverity.ERROR,
                    description="Timestamps are not in ascending order",
                    affected_rows=unsorted_positions.tolist(),
                    can_fix=True,
                    fix_suggestion="Sort data by timestamp"
                ))
        
        # Vérifier les duplications
        if not self.config['allow_duplicates']:
            duplicated = data.index.duplicated()
            if duplicated.any():
                dup_positions = np.where(duplicated)[0]
                issues.append(ValidationIssue(
                    issue_type="duplicate_timestamps",
                    severity=ValidationSeverity.ERROR,
                    description=f"Found {duplicated.sum()} duplicate timestamps",
                    affected_rows=dup_positions.tolist(),
                    can_fix=True,
                    fix_suggestion="Remove or aggregate duplicate entries"
                ))
        
        # Vérifier les gaps temporels
        if len(data) > 1:
            time_diffs = pd.Series(data.index).diff().dropna()
            
            # Gaps trop grands
            max_gap = pd.Timedelta(seconds=self.config['max_gap_seconds'])
            large_gaps = time_diffs > max_gap
            if large_gaps.any():
                gap_positions = np.where(large_gaps)[0]
                max_gap_found = time_diffs.max()
                issues.append(ValidationIssue(
                    issue_type="large_time_gaps",
                    severity=ValidationSeverity.WARNING,
                    description=f"Found {large_gaps.sum()} gaps larger than {max_gap}. Max gap: {max_gap_found}",
                    affected_rows=gap_positions.tolist(),
                    metadata={'max_gap_seconds': max_gap_found.total_seconds()}
                ))
            
            # Gaps trop petits (possibles duplications)
            min_gap = pd.Timedelta(milliseconds=self.config['min_gap_milliseconds'])
            small_gaps = (time_diffs > pd.Timedelta(0)) & (time_diffs < min_gap)
            if small_gaps.any():
                gap_positions = np.where(small_gaps)[0]
                issues.append(ValidationIssue(
                    issue_type="suspiciously_small_gaps",
                    severity=ValidationSeverity.WARNING,
                    description=f"Found {small_gaps.sum()} gaps smaller than {min_gap}",
                    affected_rows=gap_positions.tolist()
                ))
        
        # Vérifier les heures de trading si configuré
        if self.config['business_hours_only']:
            non_business = ~data.index.to_series().apply(self._is_business_hours)
            if non_business.any():
                positions = np.where(non_business)[0]
                issues.append(ValidationIssue(
                    issue_type="non_business_hours",
                    severity=ValidationSeverity.INFO,
                    description=f"Found {non_business.sum()} entries outside business hours",
                    affected_rows=positions.tolist()
                ))
        
        return issues
    
    def _is_business_hours(self, timestamp: pd.Timestamp) -> bool:
        """Vérifie si le timestamp est dans les heures de trading"""
        # Simplification : lundi-vendredi, 9h-17h
        if timestamp.weekday() >= 5:  # Weekend
            return False
        if timestamp.hour < 9 or timestamp.hour >= 17:
            return False
        return True


class OHLCVValidator(BaseValidator):
    """Validateur pour les données OHLCV"""
    
    def _setup_default_config(self) -> None:
        defaults = {
            'max_price_change_pct': 50.0,    # 50% max change
            'min_volume': 0.0,
            'max_spread_pct': 10.0,          # 10% max spread
            'outlier_std_threshold': 5.0,     # 5 sigma pour outliers
            'require_positive_prices': True,
            'check_price_continuity': True
        }
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Valide les données OHLCV"""
        issues = []
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Vérifier les colonnes requises
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            issues.append(ValidationIssue(
                issue_type="missing_columns",
                severity=ValidationSeverity.CRITICAL,
                description=f"Missing required columns: {missing_columns}",
                affected_columns=list(missing_columns)
            ))
            return issues
        
        # Vérifier la cohérence OHLC
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            positions = np.where(invalid_ohlc)[0]
            issues.append(ValidationIssue(
                issue_type="invalid_ohlc_relationship",
                severity=ValidationSeverity.ERROR,
                description=f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships",
                affected_rows=positions.tolist(),
                affected_columns=['open', 'high', 'low', 'close']
            ))
        
        # Vérifier les prix négatifs
        if self.config['require_positive_prices']:
            negative_prices = (
                (data['open'] <= 0) |
                (data['high'] <= 0) |
                (data['low'] <= 0) |
                (data['close'] <= 0)
            )
            if negative_prices.any():
                positions = np.where(negative_prices)[0]
                issues.append(ValidationIssue(
                    issue_type="negative_prices",
                    severity=ValidationSeverity.ERROR,
                    description=f"Found {negative_prices.sum()} rows with non-positive prices",
                    affected_rows=positions.tolist()
                ))
        
        # Vérifier les volumes
        negative_volume = data['volume'] < self.config['min_volume']
        if negative_volume.any():
            positions = np.where(negative_volume)[0]
            issues.append(ValidationIssue(
                issue_type="invalid_volume",
                severity=ValidationSeverity.WARNING,
                description=f"Found {negative_volume.sum()} rows with volume below {self.config['min_volume']}",
                affected_rows=positions.tolist(),
                affected_columns=['volume']
            ))
        
        # Détecter les outliers de prix
        for col in ['open', 'high', 'low', 'close']:
            outliers = self._detect_outliers(data[col])
            if len(outliers) > 0:
                issues.append(ValidationIssue(
                    issue_type=f"price_outliers_{col}",
                    severity=ValidationSeverity.WARNING,
                    description=f"Found {len(outliers)} outliers in {col} prices",
                    affected_rows=outliers,
                    affected_columns=[col],
                    metadata={'outlier_values': data[col].iloc[outliers].tolist()}
                ))
        
        # Vérifier les changements de prix excessifs
        if self.config['check_price_continuity'] and len(data) > 1:
            price_changes = data['close'].pct_change().abs() * 100
            excessive_changes = price_changes > self.config['max_price_change_pct']
            
            if excessive_changes.any():
                positions = np.where(excessive_changes)[0]
                max_change = price_changes.max()
                issues.append(ValidationIssue(
                    issue_type="excessive_price_changes",
                    severity=ValidationSeverity.ERROR,
                    description=f"Found {excessive_changes.sum()} price changes exceeding {self.config['max_price_change_pct']}%. Max: {max_change:.1f}%",
                    affected_rows=positions.tolist(),
                    metadata={'max_change_pct': float(max_change)}
                ))
        
        # Vérifier les spreads
        spread_pct = ((data['high'] - data['low']) / data['low'] * 100)
        excessive_spreads = spread_pct > self.config['max_spread_pct']
        
        if excessive_spreads.any():
            positions = np.where(excessive_spreads)[0]
            issues.append(ValidationIssue(
                issue_type="excessive_spreads",
                severity=ValidationSeverity.WARNING,
                description=f"Found {excessive_spreads.sum()} rows with spread > {self.config['max_spread_pct']}%",
                affected_rows=positions.tolist()
            ))
        
        # Vérifier les valeurs manquantes
        missing = data[required_columns].isnull()
        if missing.any().any():
            for col in required_columns:
                if missing[col].any():
                    positions = np.where(missing[col])[0]
                    issues.append(ValidationIssue(
                        issue_type="missing_values",
                        severity=ValidationSeverity.ERROR,
                        description=f"Found {missing[col].sum()} missing values in {col}",
                        affected_rows=positions.tolist(),
                        affected_columns=[col],
                        can_fix=True,
                        fix_suggestion="Forward fill or interpolate missing values"
                    ))
        
        return issues
    
    def _detect_outliers(self, series: pd.Series) -> List[int]:
        """Détecte les outliers statistiques"""
        if len(series) < 10:
            return []
        
        # Z-score method
        z_scores = np.abs(stats.zscore(series.dropna()))
        threshold = self.config['outlier_std_threshold']
        outlier_positions = np.where(z_scores > threshold)[0]
        
        return outlier_positions.tolist()


class TickDataValidator(BaseValidator):
    """Validateur pour les données tick"""
    
    def _setup_default_config(self) -> None:
        defaults = {
            'max_spread_pct': 5.0,           # 5% max spread
            'min_tick_size': 0.00001,        # Minimum tick size
            'max_price_jump_pct': 10.0,      # 10% max jump
            'require_bid_ask': True,
            'check_trade_through': True,
            'outlier_volume_threshold': 10.0  # 10x average
        }
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Valide les données tick"""
        issues = []
        
        # Colonnes requises
        required = ['price', 'volume']
        if self.config['require_bid_ask']:
            required.extend(['bid', 'ask', 'bid_size', 'ask_size'])
        
        missing = set(required) - set(data.columns)
        if missing:
            issues.append(ValidationIssue(
                issue_type="missing_columns",
                severity=ValidationSeverity.CRITICAL,
                description=f"Missing required columns: {missing}",
                affected_columns=list(missing)
            ))
            return issues
        
        # Vérifier les spreads bid-ask
        if 'bid' in data.columns and 'ask' in data.columns:
            # Spreads négatifs
            negative_spreads = data['bid'] > data['ask']
            if negative_spreads.any():
                positions = np.where(negative_spreads)[0]
                issues.append(ValidationIssue(
                    issue_type="negative_spreads",
                    severity=ValidationSeverity.ERROR,
                    description=f"Found {negative_spreads.sum()} rows with bid > ask",
                    affected_rows=positions.tolist()
                ))
            
            # Spreads excessifs
            spread_pct = ((data['ask'] - data['bid']) / data['bid'] * 100)
            excessive = spread_pct > self.config['max_spread_pct']
            if excessive.any():
                positions = np.where(excessive)[0]
                max_spread = spread_pct.max()
                issues.append(ValidationIssue(
                    issue_type="excessive_tick_spreads",
                    severity=ValidationSeverity.WARNING,
                    description=f"Found {excessive.sum()} ticks with spread > {self.config['max_spread_pct']}%. Max: {max_spread:.1f}%",
                    affected_rows=positions.tolist(),
                    metadata={'max_spread_pct': float(max_spread)}
                ))
            
            # Trade-through detection
            if self.config['check_trade_through']:
                trade_through = (
                    (data['price'] < data['bid']) | 
                    (data['price'] > data['ask'])
                )
                if trade_through.any():
                    positions = np.where(trade_through)[0]
                    issues.append(ValidationIssue(
                        issue_type="trade_through",
                        severity=ValidationSeverity.ERROR,
                        description=f"Found {trade_through.sum()} trades outside bid-ask spread",
                        affected_rows=positions.tolist()
                    ))
        
        # Vérifier les sauts de prix
        if len(data) > 1:
            price_jumps = data['price'].pct_change().abs() * 100
            excessive_jumps = price_jumps > self.config['max_price_jump_pct']
            
            if excessive_jumps.any():
                positions = np.where(excessive_jumps)[0]
                max_jump = price_jumps.max()
                issues.append(ValidationIssue(
                    issue_type="excessive_price_jumps",
                    severity=ValidationSeverity.WARNING,
                    description=f"Found {excessive_jumps.sum()} price jumps > {self.config['max_price_jump_pct']}%. Max: {max_jump:.1f}%",
                    affected_rows=positions.tolist(),
                    metadata={'max_jump_pct': float(max_jump)}
                ))
        
        # Vérifier les volumes aberrants
        volume_mean = data['volume'].mean()
        volume_outliers = data['volume'] > (volume_mean * self.config['outlier_volume_threshold'])
        
        if volume_outliers.any():
            positions = np.where(volume_outliers)[0]
            issues.append(ValidationIssue(
                issue_type="volume_outliers",
                severity=ValidationSeverity.INFO,
                description=f"Found {volume_outliers.sum()} ticks with volume > {self.config['outlier_volume_threshold']}x average",
                affected_rows=positions.tolist(),
                metadata={'avg_volume': float(volume_mean)}
            ))
        
        # Vérifier tick size minimum
        if len(data) > 1:
            price_diffs = data['price'].diff().abs()
            price_diffs = price_diffs[price_diffs > 0]
            
            if len(price_diffs) > 0:
                min_tick = price_diffs.min()
                if min_tick < self.config['min_tick_size']:
                    issues.append(ValidationIssue(
                        issue_type="tick_size_violation",
                        severity=ValidationSeverity.WARNING,
                        description=f"Minimum tick size {min_tick} is below threshold {self.config['min_tick_size']}",
                        metadata={'min_tick_found': float(min_tick)}
                    ))
        
        return issues


class OrderBookValidator(BaseValidator):
    """Validateur pour les données de carnet d'ordres"""
    
    def _setup_default_config(self) -> None:
        defaults = {
            'max_levels': 50,
            'require_monotonic_prices': True,
            'max_spread_pct': 5.0,
            'min_depth_usd': 100.0,
            'check_price_overlap': True
        }
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def validate(self, data: pd.DataFrame) -> List[ValidationIssue]:
        """Valide les snapshots de carnet d'ordres"""
        issues = []
        
        # Vérifier les colonnes requises
        required = ['bids', 'asks']
        missing = set(required) - set(data.columns)
        if missing:
            issues.append(ValidationIssue(
                issue_type="missing_columns",
                severity=ValidationSeverity.CRITICAL,
                description=f"Missing required columns: {missing}",
                affected_columns=list(missing)
            ))
            return issues
        
        # Valider chaque snapshot
        for idx, row in data.iterrows():
            row_issues = self._validate_orderbook_snapshot(row, idx)
            issues.extend(row_issues)
        
        return issues
    
    def _validate_orderbook_snapshot(self, snapshot: pd.Series, row_idx: int) -> List[ValidationIssue]:
        """Valide un snapshot individuel"""
        issues = []
        
        try:
            # Parser les bids et asks (supposés être des listes de [prix, quantité])
            if isinstance(snapshot['bids'], str):
                bids = json.loads(snapshot['bids'])
            else:
                bids = snapshot['bids']
            
            if isinstance(snapshot['asks'], str):
                asks = json.loads(snapshot['asks'])
            else:
                asks = snapshot['asks']
            
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type="invalid_orderbook_format",
                severity=ValidationSeverity.ERROR,
                description=f"Cannot parse orderbook data: {str(e)}",
                affected_rows=[row_idx]
            ))
            return issues
        
        # Vérifier que les bids et asks ne sont pas vides
        if not bids or not asks:
            issues.append(ValidationIssue(
                issue_type="empty_orderbook",
                severity=ValidationSeverity.ERROR,
                description="Empty bids or asks",
                affected_rows=[row_idx]
            ))
            return issues
        
        # Vérifier l'ordre des prix
        if self.config['require_monotonic_prices']:
            # Bids doivent être décroissants
            bid_prices = [float(b[0]) for b in bids]
            if not all(bid_prices[i] >= bid_prices[i+1] for i in range(len(bid_prices)-1)):
                issues.append(ValidationIssue(
                    issue_type="non_monotonic_bids",
                    severity=ValidationSeverity.ERROR,
                    description="Bid prices not in descending order",
                    affected_rows=[row_idx]
                ))
            
            # Asks doivent être croissants
            ask_prices = [float(a[0]) for a in asks]
            if not all(ask_prices[i] <= ask_prices[i+1] for i in range(len(ask_prices)-1)):
                issues.append(ValidationIssue(
                    issue_type="non_monotonic_asks",
                    severity=ValidationSeverity.ERROR,
                    description="Ask prices not in ascending order",
                    affected_rows=[row_idx]
                ))
        
        # Vérifier le chevauchement bid/ask
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        
        if best_bid >= best_ask:
            issues.append(ValidationIssue(
                issue_type="crossed_orderbook",
                severity=ValidationSeverity.CRITICAL,
                description=f"Best bid ({best_bid}) >= best ask ({best_ask})",
                affected_rows=[row_idx],
                metadata={'best_bid': best_bid, 'best_ask': best_ask}
            ))
        
        # Vérifier le spread
        spread_pct = (best_ask - best_bid) / best_bid * 100
        if spread_pct > self.config['max_spread_pct']:
            issues.append(ValidationIssue(
                issue_type="excessive_orderbook_spread",
                severity=ValidationSeverity.WARNING,
                description=f"Spread {spread_pct:.2f}% exceeds maximum {self.config['max_spread_pct']}%",
                affected_rows=[row_idx],
                metadata={'spread_pct': spread_pct}
            ))
        
        # Vérifier la profondeur
        bid_depth_usd = sum(float(b[0]) * float(b[1]) for b in bids[:5])
        ask_depth_usd = sum(float(a[0]) * float(a[1]) for a in asks[:5])
        
        if bid_depth_usd < self.config['min_depth_usd'] or ask_depth_usd < self.config['min_depth_usd']:
            issues.append(ValidationIssue(
                issue_type="insufficient_depth",
                severity=ValidationSeverity.WARNING,
                description=f"Insufficient orderbook depth. Bid: ${bid_depth_usd:.2f}, Ask: ${ask_depth_usd:.2f}",
                affected_rows=[row_idx],
                metadata={'bid_depth': bid_depth_usd, 'ask_depth': ask_depth_usd}
            ))
        
        return issues


class DataQualityValidator:
    """
    Validateur principal qui orchestre tous les validators
    et produit un rapport de qualité global
    """
    
    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.alert_manager = alert_manager
        self.config = config or {}
        
        # Logger
        self.logger = get_structured_logger(
            "data_quality_validator",
            module="data.processors"
        )
        
        # Configuration par défaut
        self._setup_default_config()
        
        # Initialiser les validators
        self.temporal_validator = TemporalValidator(self.config.get('temporal', {}))
        self.ohlcv_validator = OHLCVValidator(self.config.get('ohlcv', {}))
        self.tick_validator = TickDataValidator(self.config.get('tick', {}))
        self.orderbook_validator = OrderBookValidator(self.config.get('orderbook', {}))
        
        # Statistiques
        self.validation_stats = defaultdict(int)
        
    def _setup_default_config(self) -> None:
        """Configure les paramètres par défaut"""
        defaults = {
            'auto_fix_minor_issues': True,
            'quality_thresholds': {
                DataQualityLevel.EXCELLENT.value: 0.95,
                DataQualityLevel.GOOD.value: 0.85,
                DataQualityLevel.ACCEPTABLE.value: 0.70,
                DataQualityLevel.POOR.value: 0.50
            },
            'severity_weights': {
                ValidationSeverity.CRITICAL.value: 1.0,
                ValidationSeverity.ERROR.value: 0.5,
                ValidationSeverity.WARNING.value: 0.2,
                ValidationSeverity.INFO.value: 0.05
            },
            'alert_on_critical': True,
            'min_rows_required': 10
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def validate_data(
        self,
        data: pd.DataFrame,
        data_type: DataType,
        auto_fix: bool = True
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """
        Valide les données et retourne les données corrigées avec un rapport
        
        Args:
            data: DataFrame à valider
            data_type: Type de données
            auto_fix: Appliquer les corrections automatiques
            
        Returns:
            Tuple (données_corrigées, rapport_validation)
        """
        start_time = datetime.now(timezone.utc)
        
        # Vérifications de base
        if data is None or data.empty:
            report = ValidationReport(
                data_type=data_type,
                total_rows=0,
                valid_rows=0,
                quality_score=0.0,
                quality_level=DataQualityLevel.UNUSABLE,
                issues=[ValidationIssue(
                    issue_type="empty_dataset",
                    severity=ValidationSeverity.CRITICAL,
                    description="Dataset is empty or None"
                )]
            )
            return data, report
        
        # Copie pour les corrections
        validated_data = data.copy()
        all_issues = []
        
        # Validation temporelle (toujours nécessaire)
        temporal_issues = self.temporal_validator.validate(validated_data)
        all_issues.extend(temporal_issues)
        
        # Appliquer les corrections temporelles si possible
        if auto_fix and self.config.get('auto_fix_minor_issues', True):
            validated_data = self._fix_temporal_issues(validated_data, temporal_issues)
        
        # Validation spécifique au type de données
        if data_type == DataType.OHLCV:
            specific_issues = self.ohlcv_validator.validate(validated_data)
        elif data_type == DataType.TICK:
            specific_issues = self.tick_validator.validate(validated_data)
        elif data_type == DataType.ORDER_BOOK:
            specific_issues = self.orderbook_validator.validate(validated_data)
        else:
            specific_issues = []
        
        all_issues.extend(specific_issues)
        
        # Calculer les statistiques
        statistics = self._calculate_statistics(validated_data, data_type)
        
        # Calculer le score de qualité
        quality_score = self._calculate_quality_score(
            len(validated_data),
            all_issues
        )
        
        # Déterminer le niveau de qualité
        quality_level = self._determine_quality_level(quality_score)
        
        # Compter les lignes valides
        critical_error_rows = set()
        for issue in all_issues:
            if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
                critical_error_rows.update(issue.affected_rows)
        
        valid_rows = len(validated_data) - len(critical_error_rows)
        
        # Créer le rapport
        validation_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        report = ValidationReport(
            data_type=data_type,
            total_rows=len(validated_data),
            valid_rows=valid_rows,
            quality_score=quality_score,
            quality_level=quality_level,
            issues=all_issues,
            statistics=statistics,
            validation_time_ms=validation_time
        )
        
        # Logger le résultat
        self.logger.info(
            "data_validation_completed",
            data_type=data_type.value,
            total_rows=report.total_rows,
            valid_rows=report.valid_rows,
            quality_score=f"{quality_score:.2f}",
            quality_level=quality_level.value,
            n_issues=len(all_issues),
            validation_time_ms=f"{validation_time:.1f}"
        )
        
        # Alerter si critique
        if self.config.get('alert_on_critical', True) and self.alert_manager:
            critical_issues = report.get_issues_by_severity(ValidationSeverity.CRITICAL)
            if critical_issues:
                asyncio.create_task(self._send_critical_alert(report, critical_issues))
        
        # Mettre à jour les statistiques
        self.validation_stats[data_type.value] += 1
        self.validation_stats[f"{data_type.value}_rows"] += len(data)
        self.validation_stats[f"{data_type.value}_{quality_level.value}"] += 1
        
        return validated_data, report
    
    def _fix_temporal_issues(
        self,
        data: pd.DataFrame,
        issues: List[ValidationIssue]
    ) -> pd.DataFrame:
        """Corrige automatiquement les problèmes temporels mineurs"""
        fixed_data = data.copy()
        
        for issue in issues:
            if issue.can_fix:
                if issue.issue_type == "non_monotonic_time":
                    # Trier par index temporel
                    fixed_data = fixed_data.sort_index()
                    self.logger.info("fixed_temporal_issue", issue_type="non_monotonic_time")
                
                elif issue.issue_type == "duplicate_timestamps":
                    # Agréger les duplicatas (moyenne)
                    fixed_data = fixed_data.groupby(fixed_data.index).mean()
                    self.logger.info("fixed_temporal_issue", issue_type="duplicate_timestamps")
        
        return fixed_data
    
    def _calculate_statistics(
        self,
        data: pd.DataFrame,
        data_type: DataType
    ) -> Dict[str, Any]:
        """Calcule les statistiques descriptives"""
        stats = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'time_range': {
                'start': data.index.min().isoformat() if len(data) > 0 else None,
                'end': data.index.max().isoformat() if len(data) > 0 else None,
                'duration_hours': (data.index.max() - data.index.min()).total_seconds() / 3600 if len(data) > 1 else 0
            }
        }
        
        # Statistiques spécifiques par type
        if data_type == DataType.OHLCV and 'close' in data.columns:
            stats['price_stats'] = {
                'mean': float(data['close'].mean()),
                'std': float(data['close'].std()),
                'min': float(data['close'].min()),
                'max': float(data['close'].max()),
                'volatility': float(data['close'].pct_change().std() * np.sqrt(252))
            }
            
            if 'volume' in data.columns:
                stats['volume_stats'] = {
                    'mean': float(data['volume'].mean()),
                    'total': float(data['volume'].sum()),
                    'zero_volume_pct': float((data['volume'] == 0).sum() / len(data) * 100)
                }
        
        elif data_type == DataType.TICK and 'price' in data.columns:
            time_diffs = pd.Series(data.index).diff().dt.total_seconds()
            stats['tick_stats'] = {
                'avg_time_between_ticks': float(time_diffs.mean()) if len(time_diffs) > 1 else 0,
                'ticks_per_second': 1 / float(time_diffs.mean()) if time_diffs.mean() > 0 else 0,
                'price_changes': int((data['price'].diff() != 0).sum())
            }
        
        return stats
    
    def _calculate_quality_score(
        self,
        total_rows: int,
        issues: List[ValidationIssue]
    ) -> float:
        """Calcule un score de qualité global (0-1)"""
        if total_rows == 0:
            return 0.0
        
        # Score de base = 1.0
        score = 1.0
        
        # Pénalités par issue selon la sévérité
        severity_weights = self.config.get('severity_weights', {})
        
        for issue in issues:
            weight = severity_weights.get(issue.severity.value, 0.1)
            affected_ratio = len(issue.affected_rows) / total_rows if issue.affected_rows else 0.1
            penalty = weight * affected_ratio
            score -= penalty
        
        # Assurer que le score reste entre 0 et 1
        return max(0.0, min(1.0, score))
    
    def _determine_quality_level(self, quality_score: float) -> DataQualityLevel:
        """Détermine le niveau de qualité basé sur le score"""
        thresholds = self.config.get('quality_thresholds', {})
        
        if quality_score >= thresholds.get(DataQualityLevel.EXCELLENT.value, 0.95):
            return DataQualityLevel.EXCELLENT
        elif quality_score >= thresholds.get(DataQualityLevel.GOOD.value, 0.85):
            return DataQualityLevel.GOOD
        elif quality_score >= thresholds.get(DataQualityLevel.ACCEPTABLE.value, 0.70):
            return DataQualityLevel.ACCEPTABLE
        elif quality_score >= thresholds.get(DataQualityLevel.POOR.value, 0.50):
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.UNUSABLE
    
    async def _send_critical_alert(
        self,
        report: ValidationReport,
        critical_issues: List[ValidationIssue]
    ) -> None:
        """Envoie une alerte pour les problèmes critiques"""
        if not self.alert_manager:
            return
        
        issues_summary = "\n".join([
            f"- {issue.issue_type}: {issue.description}"
            for issue in critical_issues[:5]  # Limiter à 5
        ])
        
        if len(critical_issues) > 5:
            issues_summary += f"\n... and {len(critical_issues) - 5} more critical issues"
        
        await self.alert_manager.send_alert(
            level=AlertLevel.CRITICAL,
            title=f"Critical Data Quality Issues - {report.data_type.value}",
            message=f"""
Data validation found {len(critical_issues)} critical issues:

{issues_summary}

Quality Score: {report.quality_score:.2f}
Quality Level: {report.quality_level.value}
Valid Rows: {report.valid_rows}/{report.total_rows} ({report.validity_percentage:.1f}%)
            """,
            metadata={
                'data_type': report.data_type.value,
                'quality_score': report.quality_score,
                'n_critical_issues': len(critical_issues)
            }
        )
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de validation globales"""
        stats = dict(self.validation_stats)
        
        # Calculer les taux de qualité par type
        for data_type in DataType:
            type_key = data_type.value
            total = stats.get(type_key, 0)
            
            if total > 0:
                quality_distribution = {}
                for level in DataQualityLevel:
                    count = stats.get(f"{type_key}_{level.value}", 0)
                    quality_distribution[level.value] = {
                        'count': count,
                        'percentage': count / total * 100
                    }
                
                stats[f"{type_key}_quality_distribution"] = quality_distribution
        
        return stats
    
    def create_quality_report(
        self,
        reports: List[ValidationReport]
    ) -> Dict[str, Any]:
        """Crée un rapport de qualité consolidé"""
        if not reports:
            return {}
        
        # Agréger les métriques
        total_rows = sum(r.total_rows for r in reports)
        total_valid = sum(r.valid_rows for r in reports)
        avg_quality = sum(r.quality_score * r.total_rows for r in reports) / total_rows if total_rows > 0 else 0
        
        # Distribution des issues par sévérité
        severity_counts = defaultdict(int)
        issue_types = defaultdict(int)
        
        for report in reports:
            for issue in report.issues:
                severity_counts[issue.severity.value] += 1
                issue_types[issue.issue_type] += 1
        
        # Top issues
        top_issues = sorted(
            issue_types.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'summary': {
                'n_validations': len(reports),
                'total_rows_validated': total_rows,
                'total_valid_rows': total_valid,
                'overall_validity_pct': total_valid / total_rows * 100 if total_rows > 0 else 0,
                'average_quality_score': avg_quality,
                'validation_time_ms': sum(r.validation_time_ms for r in reports)
            },
            'quality_distribution': {
                level.value: sum(1 for r in reports if r.quality_level == level)
                for level in DataQualityLevel
            },
            'severity_distribution': dict(severity_counts),
            'top_issues': [
                {'issue_type': issue, 'count': count}
                for issue, count in top_issues
            ],
            'by_data_type': {
                data_type.value: {
                    'count': sum(1 for r in reports if r.data_type == data_type),
                    'avg_quality': np.mean([r.quality_score for r in reports if r.data_type == data_type])
                    if any(r.data_type == data_type for r in reports) else 0
                }
                for data_type in DataType
            }
        }


# Factory functions

def create_data_validator(
    alert_manager: Optional[AlertManager] = None,
    config: Optional[Dict[str, Any]] = None
) -> DataQualityValidator:
    """Crée un validateur de données configuré"""
    return DataQualityValidator(
        alert_manager=alert_manager,
        config=config
    )


def quick_validate(
    data: pd.DataFrame,
    data_type: DataType = DataType.OHLCV
) -> ValidationReport:
    """Validation rapide sans configuration"""
    validator = DataQualityValidator()
    _, report = validator.validate_data(data, data_type, auto_fix=False)
    return report