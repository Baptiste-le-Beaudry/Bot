"""
Binance Data Collector Module
Collecteur de données optimisé pour le trading haute fréquence avec support
pour les données de marché en temps réel, le carnet d'ordres niveau 2,
et les données historiques pour l'entraînement des modèles DRL.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd

import aiohttp
import websockets
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException
import ccxt.async_support as ccxt

# Configuration du logger
logger = logging.getLogger(__name__)


class DataType(Enum):
    """Types de données supportés"""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    KLINES = "klines"
    DEPTH = "depth"
    AGG_TRADES = "aggTrades"


@dataclass
class MarketData:
    """Structure de données de marché standardisée"""
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
    klines: Optional[List] = None
    
    def to_dict(self) -> Dict:
        """Convertir en dictionnaire pour le stockage"""
        data = asdict(self)
        data['data_type'] = self.data_type.value
        return data


class BinanceCollector:
    """
    Collecteur de données Binance optimisé pour le trading algorithmique haute fréquence.
    Supporte la collecte en temps réel et historique avec gestion robuste des erreurs.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 testnet: bool = False, max_reconnect_attempts: int = 5):
        """
        Initialisation du collecteur
        
        Args:
            api_key: Clé API Binance (optionnelle pour données publiques)
            api_secret: Secret API Binance
            testnet: Utiliser le testnet Binance
            max_reconnect_attempts: Nombre max de tentatives de reconnexion
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # Clients asynchrones
        self.client: Optional[AsyncClient] = None
        self.socket_manager: Optional[BinanceSocketManager] = None
        self.ccxt_client: Optional[ccxt.binance] = None
        
        # État et gestion des connexions
        self.websocket_connections: Dict[str, Any] = {}
        self.active_streams: Dict[str, bool] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.reconnect_count: Dict[str, int] = {}
        
        # Buffers de données pour l'agrégation
        self.data_buffer: Dict[str, List[MarketData]] = {}
        self.buffer_size = 1000  # Taille max du buffer par symbole
        
        # Métriques de performance
        self.latency_stats: Dict[str, List[float]] = {}
        self.error_count: Dict[str, int] = {}
        self.last_update: Dict[str, float] = {}
        
    async def initialize(self):
        """Initialiser les connexions asynchrones"""
        try:
            # Client Binance principal
            if self.api_key and self.api_secret:
                self.client = await AsyncClient.create(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=self.testnet
                )
            else:
                self.client = await AsyncClient.create(testnet=self.testnet)
            
            self.socket_manager = BinanceSocketManager(self.client)
            
            # Client CCXT pour fonctionnalités avancées
            self.ccxt_client = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future' if not self.testnet else 'spot'
                }
            })
            
            logger.info("Collecteur Binance initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def close(self):
        """Fermer toutes les connexions"""
        # Fermer les websockets
        for stream_id in list(self.websocket_connections.keys()):
            await self.stop_stream(stream_id)
        
        # Fermer les clients
        if self.client:
            await self.client.close_connection()
        if self.ccxt_client:
            await self.ccxt_client.close()
        
        logger.info("Collecteur Binance fermé")
    
    async def start_ticker_stream(self, symbols: List[str], callback: Optional[Callable] = None) -> str:
        """
        Démarrer un stream de tickers pour plusieurs symboles
        
        Args:
            symbols: Liste des symboles à suivre
            callback: Fonction de callback pour traiter les données
            
        Returns:
            ID du stream
        """
        stream_id = f"ticker_{'_'.join(symbols)}_{int(time.time())}"
        
        async def ticker_handler(msg):
            try:
                data = self._parse_ticker_data(msg)
                await self._handle_data(stream_id, data, callback)
            except Exception as e:
                logger.error(f"Erreur dans ticker_handler: {e}")
                self.error_count[stream_id] = self.error_count.get(stream_id, 0) + 1
        
        # Créer le stream multiplex
        streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
        socket = self.socket_manager.multiplex_socket(streams)
        
        self.websocket_connections[stream_id] = socket
        self.active_streams[stream_id] = True
        
        # Démarrer la réception asynchrone
        asyncio.create_task(self._websocket_listener(stream_id, socket, ticker_handler))
        
        logger.info(f"Stream ticker démarré: {stream_id}")
        return stream_id
    
    async def start_orderbook_stream(self, symbol: str, depth: int = 20, 
                                   update_speed: int = 100, callback: Optional[Callable] = None) -> str:
        """
        Démarrer un stream du carnet d'ordres niveau 2
        
        Args:
            symbol: Symbole à suivre
            depth: Profondeur du carnet (5, 10, 20)
            update_speed: Vitesse de mise à jour en ms (100 ou 1000)
            callback: Fonction de callback
            
        Returns:
            ID du stream
        """
        stream_id = f"orderbook_{symbol}_{depth}_{int(time.time())}"
        
        async def orderbook_handler(msg):
            try:
                data = self._parse_orderbook_data(symbol, msg)
                await self._handle_data(stream_id, data, callback)
            except Exception as e:
                logger.error(f"Erreur dans orderbook_handler: {e}")
                self.error_count[stream_id] = self.error_count.get(stream_id, 0) + 1
        
        # Stream partiel du carnet d'ordres
        socket = self.socket_manager.depth_socket(
            symbol=symbol,
            depth=depth,
            interval=update_speed
        )
        
        self.websocket_connections[stream_id] = socket
        self.active_streams[stream_id] = True
        
        asyncio.create_task(self._websocket_listener(stream_id, socket, orderbook_handler))
        
        logger.info(f"Stream orderbook démarré: {stream_id}")
        return stream_id
    
    async def start_trades_stream(self, symbols: List[str], callback: Optional[Callable] = None) -> str:
        """
        Démarrer un stream des trades en temps réel
        
        Args:
            symbols: Liste des symboles
            callback: Fonction de callback
            
        Returns:
            ID du stream
        """
        stream_id = f"trades_{'_'.join(symbols)}_{int(time.time())}"
        
        async def trades_handler(msg):
            try:
                data = self._parse_trade_data(msg)
                await self._handle_data(stream_id, data, callback)
            except Exception as e:
                logger.error(f"Erreur dans trades_handler: {e}")
                self.error_count[stream_id] = self.error_count.get(stream_id, 0) + 1
        
        # Stream des trades
        streams = [f"{symbol.lower()}@trade" for symbol in symbols]
        socket = self.socket_manager.multiplex_socket(streams)
        
        self.websocket_connections[stream_id] = socket
        self.active_streams[stream_id] = True
        
        asyncio.create_task(self._websocket_listener(stream_id, socket, trades_handler))
        
        logger.info(f"Stream trades démarré: {stream_id}")
        return stream_id
    
    async def get_historical_klines(self, symbol: str, interval: str, 
                                  start_time: datetime, end_time: Optional[datetime] = None,
                                  limit: int = 1000) -> pd.DataFrame:
        """
        Récupérer les données historiques de chandeliers
        
        Args:
            symbol: Symbole
            interval: Intervalle (1m, 5m, 1h, 1d, etc.)
            start_time: Date de début
            end_time: Date de fin (par défaut: maintenant)
            limit: Nombre max de chandeliers par requête
            
        Returns:
            DataFrame avec OHLCV
        """
        if not end_time:
            end_time = datetime.now()
        
        klines = []
        current_start = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        while current_start < end_timestamp:
            try:
                batch = await self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=current_start,
                    end_str=end_timestamp,
                    limit=limit
                )
                
                if not batch:
                    break
                
                klines.extend(batch)
                
                # Mettre à jour le timestamp de début pour la prochaine requête
                current_start = batch[-1][0] + 1
                
                # Pause pour respecter les limites de taux
                await asyncio.sleep(0.1)
                
            except BinanceAPIException as e:
                logger.error(f"Erreur API Binance: {e}")
                if e.code == -1121:  # Symbole invalide
                    raise
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des klines: {e}")
                await asyncio.sleep(1)
        
        # Convertir en DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Conversion des types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Nettoyer les colonnes inutiles
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"Récupéré {len(df)} chandeliers pour {symbol}")
        return df
    
    async def get_orderbook_snapshot(self, symbol: str, limit: int = 100) -> Dict:
        """
        Obtenir un snapshot du carnet d'ordres
        
        Args:
            symbol: Symbole
            limit: Profondeur (5, 10, 20, 50, 100, 500, 1000, 5000)
            
        Returns:
            Dictionnaire avec bids et asks
        """
        try:
            orderbook = await self.client.get_order_book(symbol=symbol, limit=limit)
            
            # Formater les données
            formatted_orderbook = {
                'symbol': symbol,
                'timestamp': int(time.time() * 1000),
                'bids': [[float(price), float(qty)] for price, qty in orderbook['bids']],
                'asks': [[float(price), float(qty)] for price, qty in orderbook['asks']],
                'lastUpdateId': orderbook.get('lastUpdateId', 0)
            }
            
            return formatted_orderbook
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du carnet d'ordres: {e}")
            raise
    
    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Récupérer les trades récents
        
        Args:
            symbol: Symbole
            limit: Nombre de trades (max 1000)
            
        Returns:
            Liste des trades récents
        """
        try:
            trades = await self.client.get_recent_trades(symbol=symbol, limit=limit)
            
            # Formater les trades
            formatted_trades = []
            for trade in trades:
                formatted_trades.append({
                    'id': trade['id'],
                    'price': float(trade['price']),
                    'quantity': float(trade['qty']),
                    'timestamp': trade['time'],
                    'is_buyer_maker': trade['isBuyerMaker']
                })
            
            return formatted_trades
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des trades: {e}")
            raise
    
    async def get_exchange_info(self) -> Dict:
        """Obtenir les informations de l'exchange (symboles, limites, etc.)"""
        try:
            info = await self.client.get_exchange_info()
            return info
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos exchange: {e}")
            raise
    
    async def get_24h_ticker(self, symbol: Optional[str] = None) -> Dict:
        """Obtenir les statistiques 24h d'un ou tous les symboles"""
        try:
            if symbol:
                ticker = await self.client.get_ticker(symbol=symbol)
            else:
                ticker = await self.client.get_ticker()
            return ticker
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du ticker 24h: {e}")
            raise
    
    # Méthodes privées pour le traitement des données
    
    def _parse_ticker_data(self, msg: Dict) -> MarketData:
        """Parser les données de ticker"""
        data = msg.get('data', msg)
        
        return MarketData(
            symbol=data.get('s', ''),
            timestamp=data.get('E', int(time.time() * 1000)),
            data_type=DataType.TICKER,
            bid=float(data.get('b', 0)),
            ask=float(data.get('a', 0)),
            bid_volume=float(data.get('B', 0)),
            ask_volume=float(data.get('A', 0)),
            last_price=float(data.get('c', 0)),
            volume_24h=float(data.get('v', 0))
        )
    
    def _parse_orderbook_data(self, symbol: str, msg: Dict) -> MarketData:
        """Parser les données du carnet d'ordres"""
        data = msg.get('data', msg)
        
        orderbook = {
            'bids': [[float(p), float(q)] for p, q in data.get('bids', [])],
            'asks': [[float(p), float(q)] for p, q in data.get('asks', [])],
            'lastUpdateId': data.get('lastUpdateId', 0)
        }
        
        # Calculer le meilleur bid/ask
        best_bid = orderbook['bids'][0] if orderbook['bids'] else [0, 0]
        best_ask = orderbook['asks'][0] if orderbook['asks'] else [0, 0]
        
        return MarketData(
            symbol=symbol,
            timestamp=int(time.time() * 1000),
            data_type=DataType.ORDERBOOK,
            bid=best_bid[0],
            ask=best_ask[0],
            bid_volume=best_bid[1],
            ask_volume=best_ask[1],
            orderbook=orderbook
        )
    
    def _parse_trade_data(self, msg: Dict) -> MarketData:
        """Parser les données de trade"""
        data = msg.get('data', msg)
        
        trade_info = {
            'id': data.get('t'),
            'price': float(data.get('p', 0)),
            'quantity': float(data.get('q', 0)),
            'timestamp': data.get('T', int(time.time() * 1000)),
            'is_buyer_maker': data.get('m', False)
        }
        
        return MarketData(
            symbol=data.get('s', ''),
            timestamp=trade_info['timestamp'],
            data_type=DataType.TRADES,
            last_price=trade_info['price'],
            trades=[trade_info]
        )
    
    async def _handle_data(self, stream_id: str, data: MarketData, callback: Optional[Callable] = None):
        """Gérer les données reçues"""
        # Calculer la latence
        current_time = time.time() * 1000
        latency = current_time - data.timestamp
        
        if stream_id not in self.latency_stats:
            self.latency_stats[stream_id] = []
        self.latency_stats[stream_id].append(latency)
        
        # Limiter la taille des statistiques
        if len(self.latency_stats[stream_id]) > 1000:
            self.latency_stats[stream_id] = self.latency_stats[stream_id][-1000:]
        
        # Mettre à jour le timestamp de dernière mise à jour
        self.last_update[stream_id] = current_time
        
        # Ajouter au buffer
        symbol = data.symbol
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []
        
        self.data_buffer[symbol].append(data)
        
        # Limiter la taille du buffer
        if len(self.data_buffer[symbol]) > self.buffer_size:
            self.data_buffer[symbol] = self.data_buffer[symbol][-self.buffer_size:]
        
        # Appeler le callback si fourni
        if callback:
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"Erreur dans le callback: {e}")
    
    async def _websocket_listener(self, stream_id: str, socket, handler: Callable):
        """Écouteur websocket avec reconnexion automatique"""
        self.reconnect_count[stream_id] = 0
        
        while self.active_streams.get(stream_id, False):
            try:
                async with socket as ws:
                    logger.info(f"Websocket connecté: {stream_id}")
                    self.reconnect_count[stream_id] = 0
                    
                    async for msg in ws:
                        if not self.active_streams.get(stream_id, False):
                            break
                        
                        await handler(msg)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Connexion websocket fermée: {stream_id}")
            except Exception as e:
                logger.error(f"Erreur websocket {stream_id}: {e}")
            
            # Tentative de reconnexion
            if self.active_streams.get(stream_id, False):
                self.reconnect_count[stream_id] += 1
                
                if self.reconnect_count[stream_id] > self.max_reconnect_attempts:
                    logger.error(f"Nombre max de reconnexions atteint pour {stream_id}")
                    self.active_streams[stream_id] = False
                    break
                
                wait_time = min(2 ** self.reconnect_count[stream_id], 60)
                logger.info(f"Reconnexion dans {wait_time}s pour {stream_id}")
                await asyncio.sleep(wait_time)
    
    async def stop_stream(self, stream_id: str):
        """Arrêter un stream spécifique"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id] = False
            logger.info(f"Stream arrêté: {stream_id}")
    
    def get_buffer_data(self, symbol: str, limit: Optional[int] = None) -> List[MarketData]:
        """Récupérer les données du buffer pour un symbole"""
        if symbol not in self.data_buffer:
            return []
        
        data = self.data_buffer[symbol]
        if limit:
            return data[-limit:]
        return data.copy()
    
    def get_latency_stats(self, stream_id: str) -> Dict[str, float]:
        """Obtenir les statistiques de latence pour un stream"""
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
        
        latencies = np.array(self.latency_stats[stream_id])
        
        return {
            'mean': np.mean(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'std': np.std(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
    
    def get_stream_health(self) -> Dict[str, Dict]:
        """Obtenir l'état de santé de tous les streams"""
        health = {}
        current_time = time.time() * 1000
        
        for stream_id in self.active_streams:
            last_update = self.last_update.get(stream_id, 0)
            time_since_update = (current_time - last_update) / 1000  # en secondes
            
            health[stream_id] = {
                'active': self.active_streams.get(stream_id, False),
                'last_update_seconds_ago': time_since_update,
                'reconnect_count': self.reconnect_count.get(stream_id, 0),
                'error_count': self.error_count.get(stream_id, 0),
                'latency_stats': self.get_latency_stats(stream_id)
            }
        
        return health


# Exemple d'utilisation
async def main():
    """Exemple d'utilisation du collecteur Binance"""
    collector = BinanceCollector()
    
    try:
        # Initialiser le collecteur
        await collector.initialize()
        
        # Callback pour traiter les données
        async def handle_ticker(data: MarketData):
            logger.info(f"Ticker reçu: {data.symbol} - Bid: {data.bid}, Ask: {data.ask}")
        
        # Démarrer un stream de tickers
        symbols = ['BTCUSDT', 'ETHUSDT']
        stream_id = await collector.start_ticker_stream(symbols, handle_ticker)
        
        # Récupérer des données historiques
        start_time = datetime.now() - timedelta(days=7)
        df = await collector.get_historical_klines('BTCUSDT', '1h', start_time)
        logger.info(f"Données historiques récupérées: {len(df)} lignes")
        
        # Laisser tourner pendant 30 secondes
        await asyncio.sleep(30)
        
        # Afficher les statistiques
        health = collector.get_stream_health()
        logger.info(f"État des streams: {json.dumps(health, indent=2)}")
        
    finally:
        # Fermer proprement
        await collector.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())