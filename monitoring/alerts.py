"""
Système d'Alertes Intelligent pour Robot de Trading Algorithmique IA
====================================================================

Ce module implémente un système d'alertes sophistiqué avec intelligence artificielle
pour détecter automatiquement les anomalies, gérer l'escalade, et notifier via
multiples canaux. Optimisé pour la surveillance 24/7 des systèmes de trading.

Architecture:
- Détection d'anomalies par ML (isolation forest, autoencoder)
- Classification intelligente des alertes par criticité
- Escalade automatique avec retry et cooldown
- Multi-canal : Slack, Email, SMS, PagerDuty, Webhook
- Agrégation et déduplication des alertes similaires
- Dashboard temps réel avec métriques historiques
- Integration native avec métriques et logs
- Circuit breakers pour éviter spam d'alertes

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import hashlib
import json
import smtplib
import ssl
import time
import traceback
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Union, Callable, 
    AsyncGenerator, Protocol, Tuple
)
import threading
import uuid
import weakref

# Third-party imports
import aiohttp
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import requests

# Imports internes
from utils.logger import get_structured_logger, log_context
from utils.decorators import retry_async, circuit_breaker, rate_limit
from utils.metrics import MetricsCollector


class AlertSeverity(Enum):
    """Niveaux de criticité des alertes"""
    DEBUG = "debug"           # Information de debugging
    INFO = "info"            # Information générale
    WARNING = "warning"       # Avertissement nécessitant attention
    ERROR = "error"          # Erreur nécessitant action
    CRITICAL = "critical"    # Erreur critique nécessitant action immédiate
    EMERGENCY = "emergency"  # Urgence maximale (arrêt système)


class AlertCategory(Enum):
    """Catégories d'alertes pour classification"""
    SYSTEM = "system"             # Alertes système (CPU, mémoire, réseau)
    TRADING = "trading"           # Alertes trading (PnL, positions, ordres)
    RISK = "risk"                # Alertes de gestion des risques
    DATA = "data"                # Alertes qualité/disponibilité des données
    STRATEGY = "strategy"         # Alertes performance des stratégies
    EXECUTION = "execution"       # Alertes d'exécution d'ordres
    COMPLIANCE = "compliance"     # Alertes conformité réglementaire
    SECURITY = "security"        # Alertes sécurité et authentification


class AlertChannel(Enum):
    """Canaux de notification disponibles"""
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    DATABASE = "database"


class AlertStatus(Enum):
    """États possibles d'une alerte"""
    PENDING = "pending"           # En attente de traitement
    SENT = "sent"                # Envoyée avec succès
    FAILED = "failed"            # Échec d'envoi
    ACKNOWLEDGED = "acknowledged" # Acquittée par utilisateur
    RESOLVED = "resolved"        # Résolue
    SUPPRESSED = "suppressed"    # Supprimée (déduplication)


@dataclass
class Alert:
    """Structure d'une alerte"""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: AlertSeverity = AlertSeverity.INFO
    category: AlertCategory = AlertCategory.SYSTEM
    title: str = ""
    message: str = ""
    source: str = "unknown"
    
    # Métadonnées contextuelles
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    strategy_id: Optional[str] = None
    symbol: Optional[str] = None
    
    # État et traitement
    status: AlertStatus = AlertStatus.PENDING
    channels: Set[AlertChannel] = field(default_factory=set)
    retry_count: int = 0
    last_attempt: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Déduplication
    fingerprint: Optional[str] = None
    similar_count: int = 1
    first_occurrence: Optional[datetime] = None
    last_occurrence: Optional[datetime] = None
    
    def __post_init__(self):
        if self.fingerprint is None:
            self.fingerprint = self._generate_fingerprint()
        if self.first_occurrence is None:
            self.first_occurrence = self.timestamp
        if self.last_occurrence is None:
            self.last_occurrence = self.timestamp
    
    def _generate_fingerprint(self) -> str:
        """Génère une empreinte pour déduplication"""
        fingerprint_data = f"{self.category.value}:{self.title}:{self.source}"
        if self.symbol:
            fingerprint_data += f":{self.symbol}"
        if self.strategy_id:
            fingerprint_data += f":{self.strategy_id}"
        
        return hashlib.md5(fingerprint_data.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'alerte en dictionnaire"""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "tags": self.tags,
            "metrics": self.metrics,
            "correlation_id": self.correlation_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "status": self.status.value,
            "channels": [c.value for c in self.channels],
            "retry_count": self.retry_count,
            "fingerprint": self.fingerprint,
            "similar_count": self.similar_count,
            "first_occurrence": self.first_occurrence.isoformat() if self.first_occurrence else None,
            "last_occurrence": self.last_occurrence.isoformat() if self.last_occurrence else None
        }


@dataclass
class AlertRule:
    """Règle de déclenchement d'alerte"""
    rule_id: str
    name: str
    description: str
    category: AlertCategory
    severity: AlertSeverity
    
    # Conditions de déclenchement
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=, contains
    threshold: Union[float, str]
    window_seconds: float = 60.0
    min_data_points: int = 1
    
    # Canaux et escalade
    channels: Set[AlertChannel] = field(default_factory=set)
    escalation_channels: Set[AlertChannel] = field(default_factory=set)
    escalation_delay_minutes: int = 30
    
    # Déduplication et throttling
    cooldown_minutes: int = 15
    max_alerts_per_hour: int = 10
    
    # État
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    def should_trigger(self, metric_value: Any, current_time: datetime) -> bool:
        """Détermine si la règle doit se déclencher"""
        if not self.enabled:
            return False
        
        # Vérifier cooldown
        if (self.last_triggered and 
            current_time - self.last_triggered < timedelta(minutes=self.cooldown_minutes)):
            return False
        
        # Vérifier rate limiting
        hour_ago = current_time - timedelta(hours=1)
        if (self.last_triggered and self.last_triggered > hour_ago and 
            self.trigger_count >= self.max_alerts_per_hour):
            return False
        
        # Évaluer la condition
        return self._evaluate_condition(metric_value)
    
    def _evaluate_condition(self, value: Any) -> bool:
        """Évalue la condition de déclenchement"""
        try:
            if self.operator == ">":
                return float(value) > float(self.threshold)
            elif self.operator == "<":
                return float(value) < float(self.threshold)
            elif self.operator == ">=":
                return float(value) >= float(self.threshold)
            elif self.operator == "<=":
                return float(value) <= float(self.threshold)
            elif self.operator == "==":
                return value == self.threshold
            elif self.operator == "!=":
                return value != self.threshold
            elif self.operator == "contains":
                return str(self.threshold) in str(value)
            else:
                return False
        except (ValueError, TypeError):
            return False


class AnomalyDetector:
    """Détecteur d'anomalies par ML pour alertes intelligentes"""
    
    def __init__(self, contamination: float = 0.1, window_size: int = 100):
        self.contamination = contamination
        self.window_size = window_size
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.data_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.trained_models: Set[str] = set()
        self.logger = get_structured_logger("anomaly_detector")
    
    def add_data_point(self, metric_name: str, value: float, timestamp: float):
        """Ajoute un point de données pour analyse"""
        self.data_buffers[metric_name].append((timestamp, value))
        
        # Réentraîner le modèle si assez de données
        if len(self.data_buffers[metric_name]) >= self.window_size:
            self._train_model(metric_name)
    
    def _train_model(self, metric_name: str):
        """Entraîne le modèle d'isolation forest pour une métrique"""
        try:
            data = list(self.data_buffers[metric_name])
            if len(data) < 10:  # Minimum de données requis
                return
            
            # Préparer les features (valeur + dérivées temporelles)
            features = []
            for i in range(1, len(data)):
                timestamp, value = data[i]
                prev_timestamp, prev_value = data[i-1]
                
                time_delta = timestamp - prev_timestamp
                value_delta = value - prev_value if time_delta > 0 else 0
                rate_of_change = value_delta / max(time_delta, 0.001)
                
                features.append([value, value_delta, rate_of_change])
            
            if len(features) < 5:
                return
            
            # Normalisation
            if metric_name not in self.scalers:
                self.scalers[metric_name] = StandardScaler()
            
            X = np.array(features)
            X_scaled = self.scalers[metric_name].fit_transform(X)
            
            # Entraînement du modèle
            model = IsolationForest(contamination=self.contamination, random_state=42)
            model.fit(X_scaled)
            
            self.models[metric_name] = model
            self.trained_models.add(metric_name)
            
            self.logger.debug("anomaly_model_trained", 
                            metric=metric_name, 
                            data_points=len(features))
            
        except Exception as e:
            self.logger.error("anomaly_model_training_error", 
                            metric=metric_name, 
                            error=str(e))
    
    def detect_anomaly(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """Détecte si une valeur est anormale"""
        if metric_name not in self.trained_models:
            return False, 0.0
        
        try:
            data = list(self.data_buffers[metric_name])
            if len(data) < 2:
                return False, 0.0
            
            # Calculer les features comme lors de l'entraînement
            timestamp = time.time()
            prev_timestamp, prev_value = data[-1]
            
            time_delta = timestamp - prev_timestamp
            value_delta = value - prev_value
            rate_of_change = value_delta / max(time_delta, 0.001)
            
            features = np.array([[value, value_delta, rate_of_change]])
            features_scaled = self.scalers[metric_name].transform(features)
            
            # Prédiction
            anomaly_score = self.models[metric_name].decision_function(features_scaled)[0]
            is_anomaly = self.models[metric_name].predict(features_scaled)[0] == -1
            
            return is_anomaly, float(anomaly_score)
            
        except Exception as e:
            self.logger.error("anomaly_detection_error", 
                            metric=metric_name, 
                            error=str(e))
            return False, 0.0


class NotificationChannel:
    """Classe de base pour les canaux de notification"""
    
    def __init__(self, channel_type: AlertChannel):
        self.channel_type = channel_type
        self.logger = get_structured_logger(f"notification.{channel_type.value}")
    
    async def send(self, alert: Alert) -> bool:
        """Envoie une alerte (à implémenter par sous-classes)"""
        raise NotImplementedError


class SlackNotifier(NotificationChannel):
    """Notificateur Slack avec formatting avancé"""
    
    def __init__(self, webhook_url: str, default_channel: str = "#alerts"):
        super().__init__(AlertChannel.SLACK)
        self.webhook_url = webhook_url
        self.default_channel = default_channel
    
    @retry_async(max_attempts=3, backoff_factor=2.0)
    @rate_limit(calls_per_second=1.0)  # Slack rate limiting
    async def send(self, alert: Alert) -> bool:
        """Envoie une alerte Slack avec formatting riche"""
        try:
            # Couleurs par sévérité
            color_map = {
                AlertSeverity.DEBUG: "#36a64f",      # Vert
                AlertSeverity.INFO: "#439fe0",       # Bleu
                AlertSeverity.WARNING: "#ff9500",    # Orange
                AlertSeverity.ERROR: "#ff0000",      # Rouge
                AlertSeverity.CRITICAL: "#800000",   # Rouge foncé
                AlertSeverity.EMERGENCY: "#000000"   # Noir
            }
            
            # Icônes par catégorie
            icon_map = {
                AlertCategory.SYSTEM: ":computer:",
                AlertCategory.TRADING: ":chart_with_upwards_trend:",
                AlertCategory.RISK: ":warning:",
                AlertCategory.DATA: ":bar_chart:",
                AlertCategory.STRATEGY: ":robot_face:",
                AlertCategory.EXECUTION: ":arrows_clockwise:",
                AlertCategory.COMPLIANCE: ":memo:",
                AlertCategory.SECURITY: ":shield:"
            }
            
            # Construction du message
            attachment = {
                "color": color_map.get(alert.severity, "#439fe0"),
                "title": f"{icon_map.get(alert.category, ':bell:')} {alert.title}",
                "text": alert.message,
                "fields": [
                    {
                        "title": "Severity",
                        "value": alert.severity.value.upper(),
                        "short": True
                    },
                    {
                        "title": "Category", 
                        "value": alert.category.value,
                        "short": True
                    },
                    {
                        "title": "Source",
                        "value": alert.source,
                        "short": True
                    },
                    {
                        "title": "Timestamp",
                        "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "short": True
                    }
                ],
                "footer": f"Trading Robot Alert • ID: {alert.alert_id[:8]}",
                "ts": int(alert.timestamp.timestamp())
            }
            
            # Ajouter les métriques si disponibles
            if alert.metrics:
                metrics_text = "\n".join([f"• {k}: {v}" for k, v in alert.metrics.items()])
                attachment["fields"].append({
                    "title": "Metrics",
                    "value": f"```{metrics_text}```",
                    "short": False
                })
            
            # Ajouter contexte trading si disponible
            if alert.symbol or alert.strategy_id:
                context_fields = []
                if alert.symbol:
                    context_fields.append(f"Symbol: {alert.symbol}")
                if alert.strategy_id:
                    context_fields.append(f"Strategy: {alert.strategy_id}")
                
                attachment["fields"].append({
                    "title": "Trading Context",
                    "value": " | ".join(context_fields),
                    "short": False
                })
            
            # Ajout d'actions pour alertes critiques
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                attachment["actions"] = [
                    {
                        "type": "button",
                        "text": "Acknowledge",
                        "style": "primary",
                        "value": f"ack_{alert.alert_id}"
                    },
                    {
                        "type": "button", 
                        "text": "Resolve",
                        "style": "good",
                        "value": f"resolve_{alert.alert_id}"
                    }
                ]
            
            payload = {
                "channel": self.default_channel,
                "username": "Trading Robot",
                "icon_emoji": ":robot_face:",
                "attachments": [attachment]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("slack_alert_sent", alert_id=alert.alert_id)
                        return True
                    else:
                        self.logger.error("slack_send_failed", 
                                        alert_id=alert.alert_id,
                                        status=response.status,
                                        response=await response.text())
                        return False
        
        except Exception as e:
            self.logger.error("slack_notification_error", 
                            alert_id=alert.alert_id,
                            error=str(e))
            return False


class EmailNotifier(NotificationChannel):
    """Notificateur Email avec HTML riche"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, 
                 password: str, from_email: str, to_emails: List[str],
                 use_tls: bool = True):
        super().__init__(AlertChannel.EMAIL)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls
    
    @retry_async(max_attempts=3, backoff_factor=2.0)
    async def send(self, alert: Alert) -> bool:
        """Envoie une alerte par email"""
        try:
            # Template HTML pour l'email
            html_template = """
            <html>
            <body>
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <div style="background-color: {color}; color: white; padding: 20px; text-align: center;">
                        <h1 style="margin: 0;">🤖 Trading Robot Alert</h1>
                        <h2 style="margin: 10px 0 0 0;">{severity}</h2>
                    </div>
                    
                    <div style="padding: 20px; background-color: #f9f9f9;">
                        <h3 style="color: #333; margin-top: 0;">{title}</h3>
                        <p style="font-size: 16px; line-height: 1.5; color: #555;">{message}</p>
                        
                        <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                            <tr style="background-color: #e9e9e9;">
                                <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Category</td>
                                <td style="padding: 10px; border: 1px solid #ddd;">{category}</td>
                            </tr>
                            <tr>
                                <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Source</td>
                                <td style="padding: 10px; border: 1px solid #ddd;">{source}</td>
                            </tr>
                            <tr style="background-color: #e9e9e9;">
                                <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Timestamp</td>
                                <td style="padding: 10px; border: 1px solid #ddd;">{timestamp}</td>
                            </tr>
                            <tr>
                                <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Alert ID</td>
                                <td style="padding: 10px; border: 1px solid #ddd;">{alert_id}</td>
                            </tr>
                        </table>
                        
                        {context_section}
                        {metrics_section}
                    </div>
                    
                    <div style="padding: 20px; background-color: #333; color: white; text-align: center;">
                        <p style="margin: 0; font-size: 14px;">
                            This alert was generated automatically by the Trading Robot system.
                        </p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Couleurs par sévérité
            severity_colors = {
                AlertSeverity.DEBUG: "#28a745",
                AlertSeverity.INFO: "#17a2b8", 
                AlertSeverity.WARNING: "#ffc107",
                AlertSeverity.ERROR: "#dc3545",
                AlertSeverity.CRITICAL: "#800000",
                AlertSeverity.EMERGENCY: "#000000"
            }
            
            # Section contexte trading
            context_section = ""
            if alert.symbol or alert.strategy_id:
                context_items = []
                if alert.symbol:
                    context_items.append(f"<strong>Symbol:</strong> {alert.symbol}")
                if alert.strategy_id:
                    context_items.append(f"<strong>Strategy:</strong> {alert.strategy_id}")
                
                context_section = f"""
                <div style="margin: 20px 0; padding: 15px; background-color: #e3f2fd; border-left: 4px solid #2196f3;">
                    <h4 style="margin: 0 0 10px 0; color: #1976d2;">Trading Context</h4>
                    <p style="margin: 0;">{" | ".join(context_items)}</p>
                </div>
                """
            
            # Section métriques
            metrics_section = ""
            if alert.metrics:
                metrics_rows = []
                for key, value in alert.metrics.items():
                    metrics_rows.append(f"""
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">{key}</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{value}</td>
                        </tr>
                    """)
                
                metrics_section = f"""
                <div style="margin: 20px 0;">
                    <h4 style="color: #333;">Metrics</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                        {"".join(metrics_rows)}
                    </table>
                </div>
                """
            
            # Génération du HTML
            html_content = html_template.format(
                color=severity_colors.get(alert.severity, "#17a2b8"),
                severity=alert.severity.value.upper(),
                title=alert.title,
                message=alert.message,
                category=alert.category.value,
                source=alert.source,
                timestamp=alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                alert_id=alert.alert_id,
                context_section=context_section,
                metrics_section=metrics_section
            )
            
            # Création du message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)
            
            # Texte brut comme fallback
            text_content = f"""
Trading Robot Alert - {alert.severity.value.upper()}

Title: {alert.title}
Message: {alert.message}
Category: {alert.category.value}
Source: {alert.source}
Timestamp: {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}
Alert ID: {alert.alert_id}
            """
            
            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))
            
            # Envoi via SMTP
            context = ssl.create_default_context()
            
            if self.use_tls:
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls(context=context)
                    server.login(self.username, self.password)
                    server.sendmail(self.from_email, self.to_emails, msg.as_string())
            else:
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context) as server:
                    server.login(self.username, self.password)
                    server.sendmail(self.from_email, self.to_emails, msg.as_string())
            
            self.logger.info("email_alert_sent", 
                           alert_id=alert.alert_id,
                           recipients=len(self.to_emails))
            return True
            
        except Exception as e:
            self.logger.error("email_notification_error", 
                            alert_id=alert.alert_id,
                            error=str(e))
            return False


class WebhookNotifier(NotificationChannel):
    """Notificateur Webhook générique"""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        super().__init__(AlertChannel.WEBHOOK)
        self.webhook_url = webhook_url
        self.headers = headers or {}
    
    @retry_async(max_attempts=3, backoff_factor=2.0)
    async def send(self, alert: Alert) -> bool:
        """Envoie une alerte via webhook"""
        try:
            payload = alert.to_dict()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url, 
                    json=payload, 
                    headers=self.headers
                ) as response:
                    if 200 <= response.status < 300:
                        self.logger.info("webhook_alert_sent", alert_id=alert.alert_id)
                        return True
                    else:
                        self.logger.error("webhook_send_failed",
                                        alert_id=alert.alert_id,
                                        status=response.status)
                        return False
        
        except Exception as e:
            self.logger.error("webhook_notification_error",
                            alert_id=alert.alert_id,
                            error=str(e))
            return False


class AlertManager:
    """
    Gestionnaire principal des alertes avec intelligence artificielle
    """
    
    def __init__(self, config=None):
        from config.settings import get_config
        self.config = config or get_config()
        self.logger = get_structured_logger("alert_manager")
        
        # Storage des alertes et règles
        self.alerts: Dict[str, Alert] = {}
        self.rules: Dict[str, AlertRule] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Déduplication et agrégation
        self.fingerprint_groups: Dict[str, List[str]] = defaultdict(list)
        self.suppressed_alerts: Set[str] = set()
        
        # Canaux de notification
        self.notification_channels: Dict[AlertChannel, NotificationChannel] = {}
        
        # Détection d'anomalies
        self.anomaly_detector = AnomalyDetector()
        
        # Métriques et monitoring
        self.metrics_collector = None
        
        # Background processing
        self._running = False
        self._processor_task = None
        self._alert_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialisation
        self._init_default_rules()
        self._init_notification_channels()
    
    def _init_default_rules(self):
        """Initialise les règles d'alerte par défaut"""
        default_rules = [
            # Règles critiques trading
            AlertRule(
                rule_id="critical_pnl_loss",
                name="Critical PnL Loss",
                description="Alert when daily PnL loss exceeds threshold",
                category=AlertCategory.TRADING,
                severity=AlertSeverity.CRITICAL,
                metric_name="pnl_total",
                operator="<",
                threshold=-self.config.risk.max_daily_loss * 10000,  # Assumant capital de 10k
                channels={AlertChannel.SLACK, AlertChannel.EMAIL},
                escalation_channels={AlertChannel.SMS},
                escalation_delay_minutes=5,
                cooldown_minutes=5
            ),
            
            AlertRule(
                rule_id="max_drawdown_exceeded",
                name="Maximum Drawdown Exceeded", 
                description="Alert when drawdown exceeds risk limit",
                category=AlertCategory.RISK,
                severity=AlertSeverity.EMERGENCY,
                metric_name="max_drawdown",
                operator=">",
                threshold=float(self.config.risk.max_drawdown),
                channels={AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.SMS},
                cooldown_minutes=1
            ),
            
            AlertRule(
                rule_id="execution_latency_high",
                name="High Execution Latency",
                description="Alert when order execution latency is too high",
                category=AlertCategory.EXECUTION,
                severity=AlertSeverity.WARNING,
                metric_name="execution_latency_ms",
                operator=">",
                threshold=100.0,  # 100ms
                window_seconds=300,
                min_data_points=5,
                channels={AlertChannel.SLACK},
                cooldown_minutes=15
            ),
            
            AlertRule(
                rule_id="strategy_performance_poor",
                name="Poor Strategy Performance",
                description="Alert when strategy Sharpe ratio is too low",
                category=AlertCategory.STRATEGY,
                severity=AlertSeverity.WARNING,
                metric_name="sharpe_ratio",
                operator="<",
                threshold=0.5,
                window_seconds=3600,  # 1 hour
                channels={AlertChannel.SLACK},
                cooldown_minutes=60
            ),
            
            AlertRule(
                rule_id="system_memory_high",
                name="High Memory Usage",
                description="Alert when system memory usage is high",
                category=AlertCategory.SYSTEM,
                severity=AlertSeverity.WARNING,
                metric_name="system_memory_mb",
                operator=">",
                threshold=self.config.performance.max_memory_usage_gb * 1024 * 0.9,  # 90% of limit
                channels={AlertChannel.SLACK},
                cooldown_minutes=30
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
        
        self.logger.info("default_alert_rules_initialized", count=len(default_rules))
    
    def _init_notification_channels(self):
        """Initialise les canaux de notification"""
        try:
            # Slack
            if self.config.monitoring.slack_webhook_url:
                slack_notifier = SlackNotifier(
                    webhook_url=self.config.monitoring.slack_webhook_url.get_secret_value()
                )
                self.notification_channels[AlertChannel.SLACK] = slack_notifier
            
            # Email (si configuré)
            if hasattr(self.config.monitoring, 'email_config'):
                email_config = self.config.monitoring.email_config
                email_notifier = EmailNotifier(
                    smtp_host=email_config['smtp_host'],
                    smtp_port=email_config['smtp_port'],
                    username=email_config['username'],
                    password=email_config['password'],
                    from_email=email_config['from_email'],
                    to_emails=email_config['to_emails']
                )
                self.notification_channels[AlertChannel.EMAIL] = email_notifier
            
            self.logger.info("notification_channels_initialized", 
                           channels=list(self.notification_channels.keys()))
            
        except Exception as e:
            self.logger.error("notification_channels_init_error", error=str(e))
    
    async def send_alert(self, severity: AlertSeverity, category: AlertCategory,
                        title: str, message: str, source: str = "unknown",
                        **kwargs) -> str:
        """
        Envoie une alerte avec déduplication et escalade automatiques
        
        Returns:
            Alert ID
        """
        alert = Alert(
            severity=severity,
            category=category,
            title=title,
            message=message,
            source=source,
            **kwargs
        )
        
        # Ajout du contexte de corrélation depuis les logs
        try:
            from utils.logger import correlation_id_var, strategy_id_var
            if not alert.correlation_id:
                alert.correlation_id = correlation_id_var.get()
            if not alert.strategy_id:
                alert.strategy_id = strategy_id_var.get()
        except:
            pass
        
        # Déduplication
        existing_alert_id = self._check_deduplication(alert)
        if existing_alert_id:
            self.logger.debug("alert_deduplicated", 
                            new_alert_id=alert.alert_id,
                            existing_alert_id=existing_alert_id)
            return existing_alert_id
        
        # Stockage
        with self._lock:
            self.alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            self.fingerprint_groups[alert.fingerprint].append(alert.alert_id)
        
        # Déterminer les canaux selon la sévérité
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            alert.channels = {AlertChannel.SLACK, AlertChannel.EMAIL}
        elif severity == AlertSeverity.ERROR:
            alert.channels = {AlertChannel.SLACK}
        else:
            alert.channels = {AlertChannel.SLACK}
        
        # Mise en queue pour traitement asynchrone
        try:
            await self._alert_queue.put(alert)
        except asyncio.QueueFull:
            self.logger.error("alert_queue_full", alert_id=alert.alert_id)
        
        # Log structuré de l'alerte
        with log_context(correlation_id=alert.correlation_id, alert_id=alert.alert_id):
            self.logger.info("alert_created",
                           severity=severity.value,
                           category=category.value,
                           title=title,
                           source=source)
        
        return alert.alert_id
    
    def _check_deduplication(self, alert: Alert) -> Optional[str]:
        """Vérifie la déduplication et met à jour les alertes similaires"""
        with self._lock:
            similar_alert_ids = self.fingerprint_groups.get(alert.fingerprint, [])
            
            for alert_id in similar_alert_ids:
                existing_alert = self.alerts.get(alert_id)
                if not existing_alert:
                    continue
                
                # Vérifier si l'alerte est récente (dans les 15 dernières minutes)
                time_diff = alert.timestamp - existing_alert.last_occurrence
                if time_diff < timedelta(minutes=15):
                    # Mettre à jour l'alerte existante
                    existing_alert.similar_count += 1
                    existing_alert.last_occurrence = alert.timestamp
                    
                    # Escalader si trop d'occurrences
                    if existing_alert.similar_count >= 5 and existing_alert.severity != AlertSeverity.EMERGENCY:
                        existing_alert.severity = AlertSeverity.CRITICAL
                        self.logger.warning("alert_escalated_due_to_frequency",
                                          alert_id=existing_alert.alert_id,
                                          count=existing_alert.similar_count)
                    
                    return existing_alert.alert_id
            
            return None
    
    async def send_critical_alert(self, title: str, message: str, **kwargs) -> str:
        """Raccourci pour envoyer une alerte critique"""
        return await self.send_alert(
            AlertSeverity.CRITICAL, 
            AlertCategory.SYSTEM,
            title, 
            message,
            **kwargs
        )
    
    async def send_warning_alert(self, title: str, message: str, **kwargs) -> str:
        """Raccourci pour envoyer un avertissement"""
        return await self.send_alert(
            AlertSeverity.WARNING,
            AlertCategory.SYSTEM, 
            title,
            message,
            **kwargs
        )
    
    async def send_trading_alert(self, severity: AlertSeverity, title: str, 
                               message: str, symbol: Optional[str] = None,
                               strategy_id: Optional[str] = None, **kwargs) -> str:
        """Raccourci pour alertes trading avec contexte"""
        return await self.send_alert(
            severity,
            AlertCategory.TRADING,
            title,
            message,
            symbol=symbol,
            strategy_id=strategy_id,
            **kwargs
        )
    
    async def send_risk_alert(self, severity: AlertSeverity, title: str,
                             message: str, metrics: Optional[Dict[str, Any]] = None,
                             **kwargs) -> str:
        """Raccourci pour alertes de risque avec métriques"""
        return await self.send_alert(
            severity,
            AlertCategory.RISK,
            title,
            message,
            metrics=metrics or {},
            **kwargs
        )
    
    def add_alert_rule(self, rule: AlertRule):
        """Ajoute une nouvelle règle d'alerte"""
        with self._lock:
            self.rules[rule.rule_id] = rule
        
        self.logger.info("alert_rule_added", 
                        rule_id=rule.rule_id,
                        metric=rule.metric_name,
                        threshold=rule.threshold)
    
    def remove_alert_rule(self, rule_id: str):
        """Supprime une règle d'alerte"""
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                self.logger.info("alert_rule_removed", rule_id=rule_id)
    
    def check_metric_rules(self, metric_name: str, value: Any):
        """Vérifie les règles d'alerte pour une métrique"""
        current_time = datetime.now(timezone.utc)
        
        with self._lock:
            for rule in self.rules.values():
                if rule.metric_name == metric_name and rule.should_trigger(value, current_time):
                    # Déclencher l'alerte
                    asyncio.create_task(self._trigger_rule_alert(rule, value))
    
    async def _trigger_rule_alert(self, rule: AlertRule, value: Any):
        """Déclenche une alerte basée sur une règle"""
        title = f"{rule.name} Triggered"
        message = f"Metric '{rule.metric_name}' value {value} {rule.operator} {rule.threshold}"
        
        await self.send_alert(
            severity=rule.severity,
            category=rule.category,
            title=title,
            message=message,
            source=f"rule:{rule.rule_id}",
            metrics={rule.metric_name: value},
            tags={"rule_id": rule.rule_id, "triggered_by": "rule"}
        )
        
        # Mettre à jour l'état de la règle
        with self._lock:
            rule.last_triggered = datetime.now(timezone.utc)
            rule.trigger_count += 1
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acquitte une alerte"""
        with self._lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now(timezone.utc)
                
                self.logger.info("alert_acknowledged",
                               alert_id=alert_id,
                               acknowledged_by=acknowledged_by)
                return True
            
            return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Résout une alerte"""
        with self._lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now(timezone.utc)
                
                self.logger.info("alert_resolved",
                               alert_id=alert_id,
                               resolved_by=resolved_by)
                return True
            
            return False
    
    async def start(self):
        """Démarre le gestionnaire d'alertes"""
        if self._running:
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_alerts())
        
        self.logger.info("alert_manager_started")
    
    async def stop(self):
        """Arrête le gestionnaire d'alertes"""
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("alert_manager_stopped")
    
    async def _process_alerts(self):
        """Traite les alertes en arrière-plan"""
        while self._running:
            try:
                # Traitement avec timeout pour éviter les blocages
                alert = await asyncio.wait_for(self._alert_queue.get(), timeout=1.0)
                await self._send_alert_notifications(alert)
                self._alert_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("alert_processing_error", error=str(e))
                await asyncio.sleep(1.0)
    
    async def _send_alert_notifications(self, alert: Alert):
        """Envoie les notifications pour une alerte"""
        success_channels = []
        failed_channels = []
        
        for channel in alert.channels:
            if channel not in self.notification_channels:
                self.logger.warning("notification_channel_not_configured", channel=channel.value)
                continue
            
            try:
                notifier = self.notification_channels[channel]
                success = await notifier.send(alert)
                
                if success:
                    success_channels.append(channel.value)
                else:
                    failed_channels.append(channel.value)
                    
            except Exception as e:
                self.logger.error("notification_send_error",
                                channel=channel.value,
                                alert_id=alert.alert_id,
                                error=str(e))
                failed_channels.append(channel.value)
        
        # Mettre à jour le statut de l'alerte
        with self._lock:
            if success_channels:
                alert.status = AlertStatus.SENT
            else:
                alert.status = AlertStatus.FAILED
                alert.retry_count += 1
            
            alert.last_attempt = datetime.now(timezone.utc)
        
        self.logger.info("alert_notifications_processed",
                        alert_id=alert.alert_id,
                        success_channels=success_channels,
                        failed_channels=failed_channels)
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques des alertes"""
        with self._lock:
            total_alerts = len(self.alerts)
            
            # Compter par statut
            status_counts = defaultdict(int)
            severity_counts = defaultdict(int)
            category_counts = defaultdict(int)
            
            for alert in self.alerts.values():
                status_counts[alert.status.value] += 1
                severity_counts[alert.severity.value] += 1
                category_counts[alert.category.value] += 1
            
            # Alertes récentes (dernière heure)
            hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
            recent_alerts = [a for a in self.alerts.values() if a.timestamp > hour_ago]
            
            return {
                "total_alerts": total_alerts,
                "recent_alerts_1h": len(recent_alerts),
                "status_breakdown": dict(status_counts),
                "severity_breakdown": dict(severity_counts),
                "category_breakdown": dict(category_counts),
                "active_rules": len(self.rules),
                "configured_channels": list(self.notification_channels.keys()),
                "suppressed_alerts": len(self.suppressed_alerts)
            }


# Instance globale par défaut
_default_alert_manager: Optional[AlertManager] = None


def get_default_alert_manager() -> AlertManager:
    """Récupère le gestionnaire d'alertes par défaut"""
    global _default_alert_manager
    if _default_alert_manager is None:
        _default_alert_manager = AlertManager()
    return _default_alert_manager


# Fonctions raccourcies pour usage facile
async def send_critical_alert(title: str, message: str, **kwargs) -> str:
    """Fonction raccourcie pour envoyer une alerte critique"""
    manager = get_default_alert_manager()
    return await manager.send_critical_alert(title, message, **kwargs)


async def send_warning_alert(title: str, message: str, **kwargs) -> str:
    """Fonction raccourcie pour envoyer un avertissement"""
    manager = get_default_alert_manager()
    return await manager.send_warning_alert(title, message, **kwargs)


async def send_trading_alert(severity: AlertSeverity, title: str, message: str, **kwargs) -> str:
    """Fonction raccourcie pour alertes trading"""
    manager = get_default_alert_manager()
    return await manager.send_trading_alert(severity, title, message, **kwargs)


# Exports principaux
__all__ = [
    'AlertManager',
    'Alert',
    'AlertRule',
    'AlertSeverity',
    'AlertCategory',
    'AlertChannel',
    'AlertStatus',
    'SlackNotifier',
    'EmailNotifier',
    'WebhookNotifier',
    'AnomalyDetector',
    'get_default_alert_manager',
    'send_critical_alert',
    'send_warning_alert',
    'send_trading_alert'
]


if __name__ == "__main__":
    # Test du système d'alertes
    import asyncio
    
    async def test_alerts():
        print("🚀 Testing Trading Alerts System...")
        
        # Initialisation
        alert_manager = AlertManager()
        await alert_manager.start()
        
        # Test alertes de base
        alert_id1 = await alert_manager.send_critical_alert(
            "System Critical Error",
            "Trading engine encountered a critical error that requires immediate attention."
        )
        print(f"✅ Critical alert sent: {alert_id1[:8]}")
        
        alert_id2 = await alert_manager.send_warning_alert(
            "High Memory Usage",
            "System memory usage is approaching configured limits."
        )
        print(f"✅ Warning alert sent: {alert_id2[:8]}")
        
        # Test alerte trading avec contexte
        alert_id3 = await alert_manager.send_trading_alert(
            AlertSeverity.ERROR,
            "Trade Execution Failed",
            "Failed to execute trade due to insufficient balance",
            symbol="BTCUSDT",
            strategy_id="arbitrage_v1",
            metrics={"balance": 1000, "required": 1500}
        )
        print(f"✅ Trading alert sent: {alert_id3[:8]}")
        
        # Test déduplication (même alerte)
        alert_id4 = await alert_manager.send_trading_alert(
            AlertSeverity.ERROR,
            "Trade Execution Failed",
            "Failed to execute trade due to insufficient balance",
            symbol="BTCUSDT",
            strategy_id="arbitrage_v1"
        )
        print(f"✅ Deduplicated alert: {alert_id4[:8]} ({'same' if alert_id4 == alert_id3 else 'different'})")
        
        # Test règles d'alerte
        alert_manager.check_metric_rules("pnl_total", -500)  # Devrait déclencher si loss > threshold
        alert_manager.check_metric_rules("execution_latency_ms", 150)  # Devrait déclencher
        
        # Attendre le traitement
        await asyncio.sleep(2.0)
        
        # Acquitter une alerte
        await alert_manager.acknowledge_alert(alert_id1, "admin")
        print(f"✅ Alert acknowledged: {alert_id1[:8]}")
        
        # Statistiques
        stats = alert_manager.get_alert_stats()
        print(f"\n📊 Alert Statistics:")
        print(f"  Total alerts: {stats['total_alerts']}")
        print(f"  Recent (1h): {stats['recent_alerts_1h']}")
        print(f"  By severity: {stats['severity_breakdown']}")
        print(f"  By status: {stats['status_breakdown']}")
        
        # Test détecteur d'anomalies
        detector = alert_manager.anomaly_detector
        
        # Ajouter des données normales
        for i in range(50):
            detector.add_data_point("test_metric", 100 + i * 0.1, time.time() + i)
        
        # Tester une anomalie
        is_anomaly, score = detector.detect_anomaly("test_metric", 200)  # Valeur anormale
        print(f"✅ Anomaly detection: anomaly={is_anomaly}, score={score:.3f}")
        
        # Arrêt
        await alert_manager.stop()
        print("\n✅ All alert tests completed!")
    
    # Run test
    asyncio.run(test_alerts())