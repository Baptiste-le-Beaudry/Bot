"""
Data Collectors Sub-package
==========================

Collecteurs de données pour différents exchanges et sources.
"""

from data.collectors.binance_collector import (
    BinanceCollector,
    BinanceWebSocketManager,
    BinanceRestClient,
    BinanceDataStream
)

from data.collectors.ib_collector import (
    InteractiveBrokersCollector,
    IBClient,
    IBDataStream,
    ContractDetails
)

from data.collectors.multi_exchange import (
    MultiExchangeCollector,
    ExchangeManager,
    DataAggregator,
    SymbolMapper
)

# Base classes
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
import asyncio


class CollectorState(Enum):
    """États d'un collecteur"""
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    COLLECTING = "collecting"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    STOPPED = "stopped"


class DataSource(Enum):
    """Sources de données disponibles"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    INTERACTIVE_BROKERS = "ib"
    FTX = "ftx"
    BYBIT = "bybit"


@dataclass
class StreamConfig:
    """Configuration d'un stream de données"""
    symbols: List[str]
    data_types: List[str]
    interval: Optional[str] = None
    depth: Optional[int] = None
    callbacks: Dict[str, Callable] = None


class BaseCollector(ABC):
    """Classe de base pour tous les collecteurs"""
    
    def __init__(self, config: dict):
        self.config = config
        self.state = CollectorState.IDLE
        self.streams = {}
        self.callbacks = {}
        self._tasks = []
        
    @abstractmethod
    async def connect(self):
        """Établit la connexion"""
        pass
        
    @abstractmethod
    async def disconnect(self):
        """Ferme la connexion"""
        pass
        
    @abstractmethod
    async def subscribe(self, symbols: List[str], data_types: List[str]):
        """Souscrit aux données"""
        pass
        
    @abstractmethod
    async def unsubscribe(self, symbols: List[str], data_types: List[str]):
        """Désinscrit des données"""
        pass
        
    def register_callback(self, data_type: str, callback: Callable):
        """Enregistre un callback pour un type de données"""
        if data_type not in self.callbacks:
            self.callbacks[data_type] = []
        self.callbacks[data_type].append(callback)
        
    async def start(self):
        """Démarre la collecte"""
        self.state = CollectorState.CONNECTING
        await self.connect()
        self.state = CollectorState.COLLECTING
        
    async def stop(self):
        """Arrête la collecte"""
        self.state = CollectorState.STOPPED
        for task in self._tasks:
            task.cancel()
        await self.disconnect()


__all__ = [
    # Binance
    "BinanceCollector",
    "BinanceWebSocketManager",
    "BinanceRestClient",
    "BinanceDataStream",
    
    # Interactive Brokers
    "InteractiveBrokersCollector",
    "IBClient",
    "IBDataStream",
    "ContractDetails",
    
    # Multi-Exchange
    "MultiExchangeCollector",
    "ExchangeManager",
    "DataAggregator",
    "SymbolMapper",
    
    # Base classes
    "BaseCollector",
    "CollectorState",
    "DataSource",
    "StreamConfig"
]