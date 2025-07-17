"""
Interactive Brokers Data Collector Module
Collecteur de données optimisé pour le trading algorithmique via l'API IB.
Support pour données temps réel, historiques, et carnet d'ordres niveau 2.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

from ib_insync import (
    IB, Contract, Stock, Forex, Future, Option, 
    Ticker, BarData, MarketDepth, Trade,
    util, MarketDataType
)
import nest_asyncio

# Permet l'utilisation d'asyncio dans Jupyter et environnements imbriqués
nest_asyncio.apply()

# Configuration du logger
logger = logging.getLogger(__name__)


class DataType(Enum):
    """Types de données supportés"""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    BARS = "bars"
    DEPTH = "depth"
    HISTORICAL = "historical"


class SecurityType(Enum):
    """Types d'instruments supportés"""
    STOCK = "STK"
    FOREX = "CASH"
    FUTURE = "FUT"
    OPTION = "OPT"
    INDEX = "IND"
    CRYPTO = "CRYPTO"


@dataclass
class MarketData:
    """Structure de données de marché standardisée (compatible avec binance_collector)"""
    symbol: str
    timestamp: int
    data_type: DataType
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    last_price: Optional[float] = None
    volume_24h: Optional[float] = None
    orderbook: Optional[Dict] = None
    trades: Optional[List] = None
    bars: Optional[List] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    close: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convertir en dictionnaire pour le stockage"""
        data = asdict(self)
        data['data_type'] = self.data_type.value
        return data


class IBCollector:
    """
    Collecteur de données Interactive Brokers optimisé pour le trading algorithmique.
    Compatible avec l'architecture du projet pour l'intégration avec DRL et stratégies HFT.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, 
                 client_id: int = 1, paper_trading: bool = True,
                 max_reconnect_attempts: int = 5):
        """
        Initialisation du collecteur IB
        
        Args:
            host: Adresse IP de TWS/Gateway
            port: Port de connexion (7497 pour TWS paper, 7496 pour live)
            client_id: ID unique du client
            paper_trading: Mode paper trading
            max_reconnect_attempts: Nombre max de tentatives de reconnexion
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.paper_trading = paper_trading
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # Client IB
        self.ib = IB()
        self.connected = False
        
        # Gestion des souscriptions
        self.subscriptions: Dict[str, Any] = {}
        self.active_contracts: Dict[str, Contract] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        
        # Buffers de données
        self.data_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.ticker_data: Dict[str, Ticker] = {}
        self.orderbook_data: Dict[str, List[MarketDepth]] = {}
        
        # Métriques de performance
        self.latency_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.error_count: Dict[str, int] = defaultdict(int)
        self.last_update: Dict[str, float] = {}
        
        # Thread pool pour les opérations non-async
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Configuration des handlers
        self._setup_event_handlers()
    
    async def initialize(self):
        """Initialiser la connexion à IB"""
        try:
            # Connexion à TWS/Gateway
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self.connected = True
            
            # Configuration du type de données de marché
            if self.paper_trading:
                self.ib.reqMarketDataType(MarketDataType.DELAYED)
            else:
                self.ib.reqMarketDataType(MarketDataType.REALTIME)
            
            logger.info(f"Connecté à IB TWS/Gateway - Host: {self.host}:{self.port}, "
                       f"Client ID: {self.client_id}, Paper: {self.paper_trading}")
            
            # Démarrer le monitoring de connexion
            asyncio.create_task(self._connection_monitor())
            
        except Exception as e:
            logger.error(f"Erreur lors de la connexion à IB: {e}")
            raise
    
    async def close(self):
        """Fermer toutes les connexions"""
        try:
            # Annuler toutes les souscriptions
            for req_id in list(self.subscriptions.keys()):
                await self.unsubscribe(req_id)
            
            # Déconnecter
            if self.connected:
                self.ib.disconnect()
                self.connected = False
            
            # Fermer l'executor
            self.executor.shutdown(wait=True)
            
            logger.info("Collecteur IB fermé")
            
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture: {e}")
    
    def _setup_event_handlers(self):
        """Configurer les handlers d'événements IB"""
        # Handler pour les mises à jour de ticker
        self.ib.pendingTickersEvent += self._on_ticker_update
        
        # Handler pour le carnet d'ordres
        self.ib.updateEvent += self._on_market_depth_update
        
        # Handler pour les barres en temps réel
        self.ib.barUpdateEvent += self._on_bar_update
        
        # Handler pour les erreurs
        self.ib.errorEvent += self._on_error
        
        # Handler pour la déconnexion
        self.ib.disconnectedEvent += self._on_disconnected
    
    async def create_contract(self, symbol: str, sec_type: SecurityType, 
                            exchange: str = 'SMART', currency: str = 'USD') -> Contract:
        """
        Créer un contrat IB
        
        Args:
            symbol: Symbole de l'instrument
            sec_type: Type de sécurité
            exchange: Exchange (SMART pour routage intelligent)
            currency: Devise
            
        Returns:
            Contract IB
        """
        if sec_type == SecurityType.STOCK:
            contract = Stock(symbol, exchange, currency)
        elif sec_type == SecurityType.FOREX:
            # Pour Forex, le symbole doit être sous forme de paire
            contract = Forex(symbol)
        elif sec_type == SecurityType.FUTURE:
            contract = Future(symbol, exchange=exchange, currency=currency)
        elif sec_type == SecurityType.INDEX:
            contract = Contract(
                symbol=symbol,
                secType='IND',
                exchange=exchange,
                currency=currency
            )
        else:
            raise ValueError(f"Type de sécurité non supporté: {sec_type}")
        
        # Qualifier le contrat pour obtenir tous les détails
        await self.ib.qualifyContractsAsync(contract)
        
        return contract
    
    async def subscribe_ticker(self, symbol: str, sec_type: SecurityType = SecurityType.STOCK,
                             exchange: str = 'SMART', callback: Optional[Callable] = None) -> str:
        """
        Souscrire aux données de ticker en temps réel
        
        Args:
            symbol: Symbole
            sec_type: Type d'instrument
            exchange: Exchange
            callback: Fonction de callback
            
        Returns:
            ID de souscription
        """
        try:
            # Créer le contrat
            contract = await self.create_contract(symbol, sec_type, exchange)
            
            # Générer un ID unique
            req_id = f"ticker_{symbol}_{int(time.time())}"
            
            # Demander les données de marché
            ticker = self.ib.reqMktData(
                contract,
                genericTickList='',
                snapshot=False,
                regulatorySnapshot=False
            )
            
            # Stocker les références
            self.subscriptions[req_id] = ticker
            self.active_contracts[req_id] = contract
            self.ticker_data[symbol] = ticker
            
            if callback:
                if req_id not in self.callbacks:
                    self.callbacks[req_id] = []
                self.callbacks[req_id].append(callback)
            
            logger.info(f"Souscription ticker créée: {req_id}")
            return req_id
            
        except Exception as e:
            logger.error(f"Erreur lors de la souscription ticker: {e}")
            raise
    
    async def subscribe_orderbook(self, symbol: str, sec_type: SecurityType = SecurityType.STOCK,
                                exchange: str = 'SMART', num_rows: int = 20,
                                callback: Optional[Callable] = None) -> str:
        """
        Souscrire au carnet d'ordres niveau 2
        
        Args:
            symbol: Symbole
            sec_type: Type d'instrument
            exchange: Exchange
            num_rows: Nombre de niveaux (5, 10, 20)
            callback: Fonction de callback
            
        Returns:
            ID de souscription
        """
        try:
            # Créer le contrat
            contract = await self.create_contract(symbol, sec_type, exchange)
            
            # Générer un ID unique
            req_id = f"orderbook_{symbol}_{int(time.time())}"
            
            # Demander les données de profondeur
            self.ib.reqMktDepth(
                contract,
                numRows=num_rows,
                isSmartDepth=True
            )
            
            # Initialiser le stockage du carnet d'ordres
            self.orderbook_data[symbol] = []
            
            # Stocker les références
            self.subscriptions[req_id] = contract
            self.active_contracts[req_id] = contract
            
            if callback:
                if req_id not in self.callbacks:
                    self.callbacks[req_id] = []
                self.callbacks[req_id].append(callback)
            
            logger.info(f"Souscription orderbook créée: {req_id}")
            return req_id
            
        except Exception as e:
            logger.error(f"Erreur lors de la souscription orderbook: {e}")
            raise
    
    async def subscribe_realtime_bars(self, symbol: str, sec_type: SecurityType = SecurityType.STOCK,
                                    exchange: str = 'SMART', bar_size: int = 5,
                                    what_to_show: str = 'TRADES',
                                    callback: Optional[Callable] = None) -> str:
        """
        Souscrire aux barres en temps réel
        
        Args:
            symbol: Symbole
            sec_type: Type d'instrument
            exchange: Exchange
            bar_size: Taille des barres en secondes (5 seulement pour IB)
            what_to_show: Type de données (TRADES, BID, ASK, MIDPOINT)
            callback: Fonction de callback
            
        Returns:
            ID de souscription
        """
        try:
            # Créer le contrat
            contract = await self.create_contract(symbol, sec_type, exchange)
            
            # Générer un ID unique
            req_id = f"bars_{symbol}_{int(time.time())}"
            
            # Demander les barres en temps réel
            bars = self.ib.reqRealTimeBars(
                contract,
                barSize=bar_size,
                whatToShow=what_to_show,
                useRTH=False
            )
            
            # Stocker les références
            self.subscriptions[req_id] = bars
            self.active_contracts[req_id] = contract
            
            if callback:
                if req_id not in self.callbacks:
                    self.callbacks[req_id] = []
                self.callbacks[req_id].append(callback)
            
            logger.info(f"Souscription barres temps réel créée: {req_id}")
            return req_id
            
        except Exception as e:
            logger.error(f"Erreur lors de la souscription barres: {e}")
            raise
    
    async def get_historical_data(self, symbol: str, sec_type: SecurityType = SecurityType.STOCK,
                                exchange: str = 'SMART', duration: str = '1 D',
                                bar_size: str = '1 min', what_to_show: str = 'TRADES',
                                end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Récupérer les données historiques
        
        Args:
            symbol: Symbole
            sec_type: Type d'instrument
            exchange: Exchange
            duration: Durée (ex: '1 D', '1 W', '1 M')
            bar_size: Taille des barres (ex: '1 min', '5 mins', '1 hour')
            what_to_show: Type de données
            end_time: Date/heure de fin (par défaut: maintenant)
            
        Returns:
            DataFrame avec OHLCV
        """
        try:
            # Créer le contrat
            contract = await self.create_contract(symbol, sec_type, exchange)
            
            # Date de fin
            if not end_time:
                end_time = datetime.now()
            
            # Récupérer les données historiques
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_time,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=False,
                formatDate=1
            )
            
            # Convertir en DataFrame
            if bars:
                df = util.df(bars)
                df['symbol'] = symbol
                
                # Renommer les colonnes pour correspondre au format standard
                df.rename(columns={
                    'date': 'timestamp',
                    'volume': 'volume',
                    'average': 'vwap',
                    'barCount': 'trade_count'
                }, inplace=True)
                
                # S'assurer que timestamp est en datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                
                logger.info(f"Récupéré {len(df)} barres historiques pour {symbol}")
                return df
            else:
                logger.warning(f"Aucune donnée historique pour {symbol}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données historiques: {e}")
            raise
    
    async def get_contract_details(self, symbol: str, sec_type: SecurityType = SecurityType.STOCK,
                                 exchange: str = 'SMART') -> Dict:
        """Obtenir les détails d'un contrat"""
        try:
            contract = await self.create_contract(symbol, sec_type, exchange)
            details = await self.ib.reqContractDetailsAsync(contract)
            
            if details:
                detail = details[0]
                return {
                    'symbol': symbol,
                    'long_name': detail.longName,
                    'exchange': detail.contract.exchange,
                    'currency': detail.contract.currency,
                    'min_tick': detail.minTick,
                    'trading_hours': detail.tradingHours,
                    'liquid_hours': detail.liquidHours,
                    'contract_month': getattr(detail.contract, 'lastTradeDateOrContractMonth', None)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des détails: {e}")
            raise
    
    async def unsubscribe(self, req_id: str):
        """Annuler une souscription"""
        try:
            if req_id in self.subscriptions:
                subscription = self.subscriptions[req_id]
                
                # Annuler selon le type
                if isinstance(subscription, Ticker):
                    self.ib.cancelMktData(subscription.contract)
                elif hasattr(subscription, 'contract'):
                    # Pour orderbook
                    self.ib.cancelMktDepth(subscription.contract)
                
                # Nettoyer les références
                del self.subscriptions[req_id]
                if req_id in self.active_contracts:
                    del self.active_contracts[req_id]
                if req_id in self.callbacks:
                    del self.callbacks[req_id]
                
                logger.info(f"Souscription annulée: {req_id}")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation: {e}")
    
    # Handlers d'événements
    
    def _on_ticker_update(self, tickers: List[Ticker]):
        """Handler pour les mises à jour de ticker"""
        for ticker in tickers:
            try:
                # Trouver le symbole
                symbol = ticker.contract.symbol
                
                # Créer MarketData
                data = self._ticker_to_market_data(ticker)
                
                # Ajouter au buffer
                self.data_buffer[symbol].append(data)
                
                # Calculer la latence
                self._update_latency(f"ticker_{symbol}", data.timestamp)
                
                # Appeler les callbacks
                asyncio.create_task(self._execute_callbacks(symbol, data))
                
            except Exception as e:
                logger.error(f"Erreur dans ticker update: {e}")
                self.error_count[f"ticker_{symbol}"] += 1
    
    def _on_market_depth_update(self, trade_or_depth):
        """Handler pour les mises à jour du carnet d'ordres"""
        if isinstance(trade_or_depth, MarketDepth):
            try:
                depth = trade_or_depth
                symbol = depth.contract.symbol if hasattr(depth, 'contract') else 'UNKNOWN'
                
                # Mettre à jour le carnet d'ordres local
                self._update_orderbook(symbol, depth)
                
                # Créer MarketData
                data = self._orderbook_to_market_data(symbol)
                
                # Ajouter au buffer
                self.data_buffer[symbol].append(data)
                
                # Appeler les callbacks
                asyncio.create_task(self._execute_callbacks(f"orderbook_{symbol}", data))
                
            except Exception as e:
                logger.error(f"Erreur dans depth update: {e}")
    
    def _on_bar_update(self, bars: BarData):
        """Handler pour les mises à jour de barres"""
        try:
            symbol = bars.contract.symbol if hasattr(bars, 'contract') else 'UNKNOWN'
            
            # Créer MarketData
            data = self._bar_to_market_data(symbol, bars)
            
            # Ajouter au buffer
            self.data_buffer[symbol].append(data)
            
            # Appeler les callbacks
            asyncio.create_task(self._execute_callbacks(f"bars_{symbol}", data))
            
        except Exception as e:
            logger.error(f"Erreur dans bar update: {e}")
    
    def _on_error(self, reqId: int, errorCode: int, errorString: str):
        """Handler pour les erreurs"""
        logger.error(f"Erreur IB - ReqId: {reqId}, Code: {errorCode}, Message: {errorString}")
        
        # Incrémenter le compteur d'erreurs
        self.error_count['global'] += 1
    
    def _on_disconnected(self):
        """Handler pour la déconnexion"""
        logger.warning("Déconnecté de IB TWS/Gateway")
        self.connected = False
    
    # Méthodes de conversion
    
    def _ticker_to_market_data(self, ticker: Ticker) -> MarketData:
        """Convertir un ticker IB en MarketData"""
        return MarketData(
            symbol=ticker.contract.symbol,
            timestamp=int(ticker.time.timestamp() * 1000) if ticker.time else int(time.time() * 1000),
            data_type=DataType.TICKER,
            bid=ticker.bid if not np.isnan(ticker.bid) else None,
            ask=ticker.ask if not np.isnan(ticker.ask) else None,
            bid_volume=ticker.bidSize if not np.isnan(ticker.bidSize) else None,
            ask_volume=ticker.askSize if not np.isnan(ticker.askSize) else None,
            last_price=ticker.last if not np.isnan(ticker.last) else None,
            volume_24h=ticker.volume if not np.isnan(ticker.volume) else None,
            high=ticker.high if not np.isnan(ticker.high) else None,
            low=ticker.low if not np.isnan(ticker.low) else None,
            close=ticker.close if not np.isnan(ticker.close) else None
        )
    
    def _orderbook_to_market_data(self, symbol: str) -> MarketData:
        """Convertir le carnet d'ordres en MarketData"""
        orderbook_list = self.orderbook_data.get(symbol, [])
        
        # Organiser par côté et niveau
        bids = []
        asks = []
        
        for depth in orderbook_list:
            if depth.side == 0:  # Bid
                bids.append([depth.price, depth.size])
            else:  # Ask
                asks.append([depth.price, depth.size])
        
        # Trier
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        
        # Meilleur bid/ask
        best_bid = bids[0] if bids else [None, None]
        best_ask = asks[0] if asks else [None, None]
        
        return MarketData(
            symbol=symbol,
            timestamp=int(time.time() * 1000),
            data_type=DataType.ORDERBOOK,
            bid=best_bid[0],
            ask=best_ask[0],
            bid_volume=best_bid[1],
            ask_volume=best_ask[1],
            orderbook={
                'bids': bids[:20],  # Top 20
                'asks': asks[:20],
                'timestamp': int(time.time() * 1000)
            }
        )
    
    def _bar_to_market_data(self, symbol: str, bar: BarData) -> MarketData:
        """Convertir une barre en MarketData"""
        return MarketData(
            symbol=symbol,
            timestamp=int(bar.time.timestamp() * 1000) if hasattr(bar, 'time') else int(time.time() * 1000),
            data_type=DataType.BARS,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume_24h=bar.volume,
            bars=[{
                'time': bar.time.isoformat() if hasattr(bar, 'time') else None,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'wap': bar.average if hasattr(bar, 'average') else None,
                'count': bar.barCount if hasattr(bar, 'barCount') else None
            }]
        )
    
    def _update_orderbook(self, symbol: str, depth: MarketDepth):
        """Mettre à jour le carnet d'ordres local"""
        if symbol not in self.orderbook_data:
            self.orderbook_data[symbol] = []
        
        # Trouver et mettre à jour ou ajouter
        found = False
        for i, existing in enumerate(self.orderbook_data[symbol]):
            if existing.position == depth.position and existing.side == depth.side:
                if depth.operation == 2:  # Delete
                    del self.orderbook_data[symbol][i]
                else:  # Update or Insert
                    self.orderbook_data[symbol][i] = depth
                found = True
                break
        
        if not found and depth.operation != 2:  # Insert
            self.orderbook_data[symbol].append(depth)
    
    def _update_latency(self, stream_id: str, timestamp: int):
        """Mettre à jour les statistiques de latence"""
        current_time = time.time() * 1000
        latency = current_time - timestamp
        self.latency_stats[stream_id].append(latency)
        self.last_update[stream_id] = current_time
    
    async def _execute_callbacks(self, key: str, data: MarketData):
        """Exécuter les callbacks pour une clé donnée"""
        # Chercher les callbacks par différentes clés
        possible_keys = [
            key,
            f"ticker_{data.symbol}",
            f"orderbook_{data.symbol}",
            f"bars_{data.symbol}"
        ]
        
        for k in possible_keys:
            if k in self.callbacks:
                for callback in self.callbacks[k]:
                    try:
                        await callback(data)
                    except Exception as e:
                        logger.error(f"Erreur dans callback: {e}")
    
    async def _connection_monitor(self):
        """Monitorer la connexion et reconnecter si nécessaire"""
        reconnect_count = 0
        
        while True:
            try:
                await asyncio.sleep(30)  # Vérifier toutes les 30 secondes
                
                if not self.connected and reconnect_count < self.max_reconnect_attempts:
                    logger.info("Tentative de reconnexion...")
                    reconnect_count += 1
                    
                    try:
                        await self.initialize()
                        reconnect_count = 0
                        logger.info("Reconnexion réussie")
                    except Exception as e:
                        logger.error(f"Échec de reconnexion: {e}")
                        await asyncio.sleep(min(2 ** reconnect_count, 60))
                
            except Exception as e:
                logger.error(f"Erreur dans connection monitor: {e}")
    
    # Méthodes utilitaires
    
    def get_buffer_data(self, symbol: str, limit: Optional[int] = None) -> List[MarketData]:
        """Récupérer les données du buffer pour un symbole"""
        if symbol not in self.data_buffer:
            return []
        
        data = list(self.data_buffer[symbol])
        if limit:
            return data[-limit:]
        return data
    
    def get_latency_stats(self, stream_id: str) -> Dict[str, float]:
        """Obtenir les statistiques de latence"""
        if stream_id not in self.latency_stats or not self.latency_stats[stream_id]:
            return {
                'mean': 0,
                'min': 0,
                'max': 0,
                'std': 0,
                'p50': 0,
                'p95': 0,
                'p99': 0
            }
        
        latencies = np.array(list(self.latency_stats[stream_id]))
        
        return {
            'mean': np.mean(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'std': np.std(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtenir le statut complet du système"""
        return {
            'connected': self.connected,
            'host': f"{self.host}:{self.port}",
            'client_id': self.client_id,
            'paper_trading': self.paper_trading,
            'active_subscriptions': len(self.subscriptions),
            'buffered_symbols': len(self.data_buffer),
            'total_errors': sum(self.error_count.values()),
            'subscriptions': list(self.subscriptions.keys()),
            'latency_summary': {
                stream_id: self.get_latency_stats(stream_id)
                for stream_id in self.latency_stats
            }
        }


# Exemple d'utilisation
async def main():
    """Exemple d'utilisation du collecteur IB"""
    # Configuration
    collector = IBCollector(
        host='127.0.0.1',
        port=7497,  # Port paper trading
        client_id=1,
        paper_trading=True
    )
    
    try:
        # Initialiser
        await collector.initialize()
        
        # Callback pour traiter les données
        async def handle_ticker(data: MarketData):
            logger.info(f"Ticker: {data.symbol} - Bid: {data.bid}, Ask: {data.ask}, Last: {data.last_price}")
        
        async def handle_orderbook(data: MarketData):
            logger.info(f"OrderBook: {data.symbol} - Best Bid: {data.bid} x {data.bid_volume}, "
                       f"Best Ask: {data.ask} x {data.ask_volume}")
        
        # Souscrire aux données
        ticker_id = await collector.subscribe_ticker('AAPL', callback=handle_ticker)
        orderbook_id = await collector.subscribe_orderbook('AAPL', num_rows=10, callback=handle_orderbook)
        
        # Récupérer des données historiques
        hist_data = await collector.get_historical_data(
            'AAPL',
            duration='1 D',
            bar_size='5 mins'
        )
        logger.info(f"Données historiques: {len(hist_data)} barres")
        
        # Laisser tourner
        await asyncio.sleep(60)
        
        # Afficher le statut
        status = collector.get_system_status()
        logger.info(f"Statut système: {status}")
        
    finally:
        # Fermer proprement
        await collector.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())