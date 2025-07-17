"""
Multi-Exchange Aggregator Module
Agrégateur de données multi-exchanges pour le trading algorithmique.
Combine les flux de Binance, Interactive Brokers et autres exchanges
pour permettre l'arbitrage et une vision unifiée du marché.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# Import des collecteurs
from .binance_collector import BinanceCollector, MarketData, DataType
from .ib_collector import IBCollector, SecurityType

# Configuration du logger
logger = logging.getLogger(__name__)


class Exchange(Enum):
    """Exchanges supportés"""
    BINANCE = "binance"
    INTERACTIVE_BROKERS = "ib"
    KRAKEN = "kraken"  # Pour extension future
    COINBASE = "coinbase"  # Pour extension future


@dataclass
class UnifiedMarketData(MarketData):
    """Structure de données de marché unifiée avec info d'exchange"""
    exchange: Exchange = None
    original_symbol: str = None
    fees: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convertir en dictionnaire"""
        data = super().to_dict()
        data['exchange'] = self.exchange.value if self.exchange else None
        data['original_symbol'] = self.original_symbol
        data['fees'] = self.fees
        return data


@dataclass
class ArbitrageOpportunity:
    """Opportunité d'arbitrage détectée"""
    symbol: str
    buy_exchange: Exchange
    sell_exchange: Exchange
    buy_price: float
    sell_price: float
    buy_volume: float
    sell_volume: float
    potential_profit: float
    profit_percentage: float
    timestamp: int
    fees_considered: bool = True
    execution_risk: str = "low"  # low, medium, high
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'buy_exchange': self.buy_exchange.value,
            'sell_exchange': self.sell_exchange.value,
            'buy_price': self.buy_price,
            'sell_price': self.sell_price,
            'max_volume': min(self.buy_volume, self.sell_volume),
            'potential_profit': self.potential_profit,
            'profit_percentage': self.profit_percentage,
            'timestamp': self.timestamp,
            'fees_considered': self.fees_considered,
            'execution_risk': self.execution_risk
        }


@dataclass
class ExchangeConfig:
    """Configuration pour un exchange"""
    enabled: bool = True
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = False
    fees: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    priority: int = 1  # Priorité pour l'exécution (1 = plus haute)
    
    def get_fee(self, trade_type: str = 'taker') -> float:
        """Obtenir les frais pour un type de trade"""
        return self.fees.get(trade_type, 0.001)  # 0.1% par défaut


class MultiExchangeAggregator:
    """
    Agrégateur multi-exchanges pour données de marché unifiées.
    Supporte l'arbitrage, la consolidation de liquidité et la redondance.
    """
    
    def __init__(self, config: Dict[Exchange, ExchangeConfig]):
        """
        Initialisation de l'agrégateur
        
        Args:
            config: Configuration par exchange
        """
        self.config = config
        self.collectors: Dict[Exchange, Any] = {}
        self.active_streams: Dict[str, Set[Exchange]] = defaultdict(set)
        
        # Mapping des symboles entre exchanges
        self.symbol_mapping: Dict[str, Dict[Exchange, str]] = {}
        self._init_symbol_mappings()
        
        # Buffers de données par symbole et exchange
        self.data_buffers: Dict[str, Dict[Exchange, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        
        # Cache des dernières données par symbole/exchange
        self.latest_data: Dict[str, Dict[Exchange, UnifiedMarketData]] = defaultdict(dict)
        
        # Détection d'arbitrage
        self.arbitrage_callbacks: List[Callable] = []
        self.min_profit_threshold = 0.001  # 0.1% minimum
        self.arbitrage_opportunities: deque = deque(maxlen=1000)
        
        # Métriques
        self.metrics: Dict[str, Any] = {
            'total_updates': defaultdict(int),
            'latency_stats': defaultdict(list),
            'arbitrage_detected': 0,
            'errors': defaultdict(int)
        }
        
        # État
        self.running = False
        self._tasks: List[asyncio.Task] = []
    
    def _init_symbol_mappings(self):
        """Initialiser les mappings de symboles entre exchanges"""
        # Exemples de mappings courants
        self.symbol_mapping = {
            'BTC/USD': {
                Exchange.BINANCE: 'BTCUSDT',
                Exchange.INTERACTIVE_BROKERS: 'BTC'
            },
            'ETH/USD': {
                Exchange.BINANCE: 'ETHUSDT',
                Exchange.INTERACTIVE_BROKERS: 'ETH'
            },
            'AAPL': {
                Exchange.INTERACTIVE_BROKERS: 'AAPL'
                # Pas disponible sur Binance
            }
        }
    
    async def initialize(self):
        """Initialiser tous les collecteurs configurés"""
        initialization_tasks = []
        
        # Binance
        if Exchange.BINANCE in self.config and self.config[Exchange.BINANCE].enabled:
            binance_config = self.config[Exchange.BINANCE]
            self.collectors[Exchange.BINANCE] = BinanceCollector(
                api_key=binance_config.api_key,
                api_secret=binance_config.api_secret,
                testnet=binance_config.testnet
            )
            initialization_tasks.append(self.collectors[Exchange.BINANCE].initialize())
        
        # Interactive Brokers
        if Exchange.INTERACTIVE_BROKERS in self.config and self.config[Exchange.INTERACTIVE_BROKERS].enabled:
            ib_config = self.config[Exchange.INTERACTIVE_BROKERS]
            self.collectors[Exchange.INTERACTIVE_BROKERS] = IBCollector(
                paper_trading=ib_config.testnet
            )
            initialization_tasks.append(self.collectors[Exchange.INTERACTIVE_BROKERS].initialize())
        
        # Initialiser tous les collecteurs en parallèle
        await asyncio.gather(*initialization_tasks)
        
        self.running = True
        logger.info(f"Agrégateur multi-exchanges initialisé avec {len(self.collectors)} exchanges")
    
    async def close(self):
        """Fermer tous les collecteurs"""
        self.running = False
        
        # Annuler les tâches en cours
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Fermer les collecteurs
        close_tasks = []
        for collector in self.collectors.values():
            close_tasks.append(collector.close())
        
        await asyncio.gather(*close_tasks, return_exceptions=True)
        logger.info("Agrégateur multi-exchanges fermé")
    
    async def subscribe_ticker(self, symbol: str, callback: Optional[Callable] = None,
                             exchanges: Optional[List[Exchange]] = None) -> Dict[Exchange, str]:
        """
        Souscrire aux tickers sur plusieurs exchanges
        
        Args:
            symbol: Symbole unifié (ex: 'BTC/USD')
            callback: Callback pour traiter les données
            exchanges: Liste des exchanges (par défaut: tous)
            
        Returns:
            Dict des IDs de souscription par exchange
        """
        if exchanges is None:
            exchanges = list(self.collectors.keys())
        
        subscription_ids = {}
        
        for exchange in exchanges:
            if exchange not in self.collectors:
                continue
            
            # Obtenir le symbole spécifique à l'exchange
            exchange_symbol = self._get_exchange_symbol(symbol, exchange)
            if not exchange_symbol:
                logger.warning(f"Symbole {symbol} non disponible sur {exchange}")
                continue
            
            # Créer un callback wrapper
            async def exchange_callback(data: MarketData, exc=exchange, sym=symbol):
                await self._handle_market_data(exc, sym, data, callback)
            
            try:
                # Souscrire selon l'exchange
                if exchange == Exchange.BINANCE:
                    sub_id = await self.collectors[exchange].start_ticker_stream(
                        [exchange_symbol], exchange_callback
                    )
                elif exchange == Exchange.INTERACTIVE_BROKERS:
                    sub_id = await self.collectors[exchange].subscribe_ticker(
                        exchange_symbol, callback=exchange_callback
                    )
                
                subscription_ids[exchange] = sub_id
                self.active_streams[symbol].add(exchange)
                
            except Exception as e:
                logger.error(f"Erreur souscription ticker {symbol} sur {exchange}: {e}")
                self.metrics['errors'][f"{exchange}_{symbol}"] += 1
        
        # Démarrer la détection d'arbitrage pour ce symbole
        if len(subscription_ids) > 1:
            self._tasks.append(
                asyncio.create_task(self._arbitrage_monitor(symbol))
            )
        
        return subscription_ids
    
    async def subscribe_orderbook(self, symbol: str, depth: int = 20,
                                callback: Optional[Callable] = None,
                                exchanges: Optional[List[Exchange]] = None) -> Dict[Exchange, str]:
        """
        Souscrire aux carnets d'ordres sur plusieurs exchanges
        """
        if exchanges is None:
            exchanges = list(self.collectors.keys())
        
        subscription_ids = {}
        
        for exchange in exchanges:
            if exchange not in self.collectors:
                continue
            
            exchange_symbol = self._get_exchange_symbol(symbol, exchange)
            if not exchange_symbol:
                continue
            
            async def exchange_callback(data: MarketData, exc=exchange, sym=symbol):
                await self._handle_market_data(exc, sym, data, callback)
            
            try:
                if exchange == Exchange.BINANCE:
                    sub_id = await self.collectors[exchange].start_orderbook_stream(
                        exchange_symbol, depth=depth, callback=exchange_callback
                    )
                elif exchange == Exchange.INTERACTIVE_BROKERS:
                    sub_id = await self.collectors[exchange].subscribe_orderbook(
                        exchange_symbol, num_rows=depth, callback=exchange_callback
                    )
                
                subscription_ids[exchange] = sub_id
                self.active_streams[symbol].add(exchange)
                
            except Exception as e:
                logger.error(f"Erreur souscription orderbook {symbol} sur {exchange}: {e}")
                self.metrics['errors'][f"{exchange}_{symbol}"] += 1
        
        return subscription_ids
    
    async def get_consolidated_orderbook(self, symbol: str, depth: int = 10) -> Dict:
        """
        Obtenir un carnet d'ordres consolidé de tous les exchanges
        
        Returns:
            Carnet d'ordres agrégé avec meilleurs prix
        """
        consolidated = {
            'symbol': symbol,
            'timestamp': int(time.time() * 1000),
            'bids': [],
            'asks': [],
            'exchanges': {}
        }
        
        all_bids = []
        all_asks = []
        
        # Collecter les carnets d'ordres de chaque exchange
        for exchange in self.active_streams.get(symbol, []):
            if symbol in self.latest_data and exchange in self.latest_data[symbol]:
                data = self.latest_data[symbol][exchange]
                if data.orderbook:
                    # Ajouter l'info d'exchange à chaque niveau
                    for bid in data.orderbook.get('bids', [])[:depth]:
                        all_bids.append({
                            'price': bid[0],
                            'volume': bid[1],
                            'exchange': exchange.value
                        })
                    
                    for ask in data.orderbook.get('asks', [])[:depth]:
                        all_asks.append({
                            'price': ask[0],
                            'volume': ask[1],
                            'exchange': exchange.value
                        })
                    
                    consolidated['exchanges'][exchange.value] = {
                        'best_bid': data.bid,
                        'best_ask': data.ask,
                        'timestamp': data.timestamp
                    }
        
        # Trier et limiter la profondeur
        all_bids.sort(key=lambda x: x['price'], reverse=True)
        all_asks.sort(key=lambda x: x['price'])
        
        consolidated['bids'] = all_bids[:depth]
        consolidated['asks'] = all_asks[:depth]
        
        # Calculer les métriques agrégées
        if all_bids and all_asks:
            consolidated['best_bid'] = all_bids[0]['price']
            consolidated['best_ask'] = all_asks[0]['price']
            consolidated['spread'] = all_asks[0]['price'] - all_bids[0]['price']
            consolidated['spread_percentage'] = (consolidated['spread'] / all_asks[0]['price']) * 100
        
        return consolidated
    
    async def get_historical_data(self, symbol: str, start_time: datetime,
                                end_time: Optional[datetime] = None,
                                interval: str = '1h',
                                exchanges: Optional[List[Exchange]] = None) -> Dict[Exchange, pd.DataFrame]:
        """
        Récupérer les données historiques de plusieurs exchanges
        """
        if exchanges is None:
            exchanges = list(self.collectors.keys())
        
        historical_data = {}
        tasks = []
        
        for exchange in exchanges:
            if exchange not in self.collectors:
                continue
            
            exchange_symbol = self._get_exchange_symbol(symbol, exchange)
            if not exchange_symbol:
                continue
            
            async def fetch_historical(exc=exchange, sym=exchange_symbol):
                try:
                    if exc == Exchange.BINANCE:
                        df = await self.collectors[exc].get_historical_klines(
                            sym, interval, start_time, end_time
                        )
                    elif exc == Exchange.INTERACTIVE_BROKERS:
                        # Convertir l'intervalle pour IB
                        ib_interval = self._convert_interval_to_ib(interval)
                        duration = self._calculate_duration(start_time, end_time)
                        df = await self.collectors[exc].get_historical_data(
                            sym, duration=duration, bar_size=ib_interval
                        )
                    else:
                        df = pd.DataFrame()
                    
                    return exc, df
                    
                except Exception as e:
                    logger.error(f"Erreur récupération historique {symbol} sur {exc}: {e}")
                    return exc, pd.DataFrame()
            
            tasks.append(fetch_historical())
        
        # Exécuter en parallèle
        results = await asyncio.gather(*tasks)
        
        for exchange, df in results:
            if not df.empty:
                historical_data[exchange] = df
        
        return historical_data
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity,
                              max_amount: Optional[float] = None) -> Dict[str, Any]:
        """
        Exécuter une opportunité d'arbitrage (simulation)
        
        Args:
            opportunity: Opportunité détectée
            max_amount: Montant maximum à trader
            
        Returns:
            Résultat de l'exécution
        """
        # Note: Ceci est une simulation. L'exécution réelle nécessiterait
        # l'intégration avec les modules d'exécution
        
        result = {
            'status': 'simulated',
            'opportunity': opportunity.to_dict(),
            'execution_time': datetime.now().isoformat(),
            'estimated_profit': 0,
            'actual_profit': None,
            'errors': []
        }
        
        try:
            # Calculer la quantité optimale
            max_volume = min(opportunity.buy_volume, opportunity.sell_volume)
            if max_amount:
                max_volume = min(max_volume, max_amount / opportunity.buy_price)
            
            # Simuler l'exécution
            buy_cost = max_volume * opportunity.buy_price
            sell_revenue = max_volume * opportunity.sell_price
            
            # Appliquer les frais
            buy_fee = buy_cost * self.config[opportunity.buy_exchange].get_fee()
            sell_fee = sell_revenue * self.config[opportunity.sell_exchange].get_fee()
            
            net_profit = sell_revenue - buy_cost - buy_fee - sell_fee
            
            result['estimated_profit'] = net_profit
            result['volume_traded'] = max_volume
            result['buy_cost'] = buy_cost
            result['sell_revenue'] = sell_revenue
            result['total_fees'] = buy_fee + sell_fee
            
            logger.info(f"Arbitrage simulé: {symbol} - Profit estimé: ${net_profit:.2f}")
            
        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(str(e))
            logger.error(f"Erreur exécution arbitrage: {e}")
        
        return result
    
    # Méthodes privées
    
    async def _handle_market_data(self, exchange: Exchange, symbol: str,
                                data: MarketData, callback: Optional[Callable] = None):
        """Traiter les données de marché reçues"""
        # Créer UnifiedMarketData
        unified_data = UnifiedMarketData(
            symbol=symbol,
            timestamp=data.timestamp,
            data_type=data.data_type,
            bid=data.bid,
            ask=data.ask,
            bid_volume=data.bid_volume,
            ask_volume=data.ask_volume,
            last_price=data.last_price,
            volume_24h=data.volume_24h,
            orderbook=data.orderbook,
            trades=data.trades,
            exchange=exchange,
            original_symbol=data.symbol,
            fees=self.config[exchange].get_fee()
        )
        
        # Mettre à jour les buffers et caches
        self.data_buffers[symbol][exchange].append(unified_data)
        self.latest_data[symbol][exchange] = unified_data
        
        # Métriques
        self.metrics['total_updates'][exchange] += 1
        
        # Callback utilisateur
        if callback:
            try:
                await callback(unified_data)
            except Exception as e:
                logger.error(f"Erreur dans callback utilisateur: {e}")
        
        # Vérifier les opportunités d'arbitrage
        await self._check_arbitrage(symbol)
    
    async def _check_arbitrage(self, symbol: str):
        """Vérifier les opportunités d'arbitrage pour un symbole"""
        if symbol not in self.latest_data or len(self.latest_data[symbol]) < 2:
            return
        
        exchanges_data = self.latest_data[symbol]
        current_time = int(time.time() * 1000)
        
        # Comparer toutes les paires d'exchanges
        exchanges = list(exchanges_data.keys())
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                ex1, ex2 = exchanges[i], exchanges[j]
                data1, data2 = exchanges_data[ex1], exchanges_data[ex2]
                
                # Vérifier la fraîcheur des données (max 5 secondes)
                if abs(current_time - data1.timestamp) > 5000:
                    continue
                if abs(current_time - data2.timestamp) > 5000:
                    continue
                
                # Vérifier les prix bid/ask
                if not all([data1.bid, data1.ask, data2.bid, data2.ask]):
                    continue
                
                # Détecter l'arbitrage
                opportunities = []
                
                # Cas 1: Acheter sur ex1, vendre sur ex2
                if data1.ask < data2.bid:
                    profit = data2.bid - data1.ask
                    profit_pct = (profit / data1.ask) * 100
                    
                    # Appliquer les frais
                    net_profit_pct = profit_pct - (data1.fees + data2.fees) * 100
                    
                    if net_profit_pct > self.min_profit_threshold * 100:
                        opportunities.append(ArbitrageOpportunity(
                            symbol=symbol,
                            buy_exchange=ex1,
                            sell_exchange=ex2,
                            buy_price=data1.ask,
                            sell_price=data2.bid,
                            buy_volume=data1.ask_volume or 0,
                            sell_volume=data2.bid_volume or 0,
                            potential_profit=profit,
                            profit_percentage=net_profit_pct,
                            timestamp=current_time,
                            fees_considered=True
                        ))
                
                # Cas 2: Acheter sur ex2, vendre sur ex1
                if data2.ask < data1.bid:
                    profit = data1.bid - data2.ask
                    profit_pct = (profit / data2.ask) * 100
                    net_profit_pct = profit_pct - (data2.fees + data1.fees) * 100
                    
                    if net_profit_pct > self.min_profit_threshold * 100:
                        opportunities.append(ArbitrageOpportunity(
                            symbol=symbol,
                            buy_exchange=ex2,
                            sell_exchange=ex1,
                            buy_price=data2.ask,
                            sell_price=data1.bid,
                            buy_volume=data2.ask_volume or 0,
                            sell_volume=data1.bid_volume or 0,
                            potential_profit=profit,
                            profit_percentage=net_profit_pct,
                            timestamp=current_time,
                            fees_considered=True
                        ))
                
                # Traiter les opportunités détectées
                for opp in opportunities:
                    await self._handle_arbitrage_opportunity(opp)
    
    async def _handle_arbitrage_opportunity(self, opportunity: ArbitrageOpportunity):
        """Traiter une opportunité d'arbitrage détectée"""
        # Évaluer le risque d'exécution
        if opportunity.profit_percentage > 5:
            opportunity.execution_risk = "high"  # Trop beau pour être vrai
        elif opportunity.profit_percentage > 2:
            opportunity.execution_risk = "medium"
        else:
            opportunity.execution_risk = "low"
        
        # Stocker l'opportunité
        self.arbitrage_opportunities.append(opportunity)
        self.metrics['arbitrage_detected'] += 1
        
        # Logger
        logger.info(f"Arbitrage détecté: {opportunity.symbol} "
                   f"{opportunity.buy_exchange.value} -> {opportunity.sell_exchange.value} "
                   f"Profit: {opportunity.profit_percentage:.2f}% "
                   f"Risk: {opportunity.execution_risk}")
        
        # Appeler les callbacks
        for callback in self.arbitrage_callbacks:
            try:
                await callback(opportunity)
            except Exception as e:
                logger.error(f"Erreur dans callback arbitrage: {e}")
    
    async def _arbitrage_monitor(self, symbol: str):
        """Moniteur continu d'arbitrage pour un symbole"""
        while self.running and symbol in self.active_streams:
            try:
                await asyncio.sleep(0.1)  # Vérifier toutes les 100ms
                await self._check_arbitrage(symbol)
            except Exception as e:
                logger.error(f"Erreur dans arbitrage monitor: {e}")
                await asyncio.sleep(1)
    
    def _get_exchange_symbol(self, unified_symbol: str, exchange: Exchange) -> Optional[str]:
        """Obtenir le symbole spécifique à un exchange"""
        if unified_symbol in self.symbol_mapping:
            return self.symbol_mapping[unified_symbol].get(exchange)
        
        # Tentative de conversion directe pour les cryptos
        if exchange == Exchange.BINANCE and '/' in unified_symbol:
            # Convertir BTC/USD -> BTCUSDT
            base, quote = unified_symbol.split('/')
            if quote == 'USD':
                return f"{base}USDT"
        
        return unified_symbol
    
    def _convert_interval_to_ib(self, interval: str) -> str:
        """Convertir l'intervalle au format IB"""
        conversions = {
            '1m': '1 min',
            '5m': '5 mins',
            '15m': '15 mins',
            '30m': '30 mins',
            '1h': '1 hour',
            '1d': '1 day'
        }
        return conversions.get(interval, '1 hour')
    
    def _calculate_duration(self, start_time: datetime, end_time: Optional[datetime]) -> str:
        """Calculer la durée pour IB"""
        if not end_time:
            end_time = datetime.now()
        
        delta = end_time - start_time
        days = delta.days
        
        if days <= 1:
            return "1 D"
        elif days <= 7:
            return f"{days} D"
        elif days <= 30:
            return f"{days // 7} W"
        else:
            return f"{days // 30} M"
    
    def add_arbitrage_callback(self, callback: Callable):
        """Ajouter un callback pour les opportunités d'arbitrage"""
        self.arbitrage_callbacks.append(callback)
    
    def get_latest_data(self, symbol: str, exchange: Optional[Exchange] = None) -> Union[UnifiedMarketData, Dict[Exchange, UnifiedMarketData]]:
        """Obtenir les dernières données pour un symbole"""
        if exchange:
            return self.latest_data.get(symbol, {}).get(exchange)
        return self.latest_data.get(symbol, {})
    
    def get_arbitrage_opportunities(self, min_profit: Optional[float] = None) -> List[ArbitrageOpportunity]:
        """Obtenir les opportunités d'arbitrage récentes"""
        opportunities = list(self.arbitrage_opportunities)
        
        if min_profit:
            opportunities = [o for o in opportunities if o.profit_percentage >= min_profit]
        
        return opportunities
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtenir les métriques du système"""
        return {
            'total_updates': dict(self.metrics['total_updates']),
            'arbitrage_detected': self.metrics['arbitrage_detected'],
            'active_symbols': len(self.active_streams),
            'active_exchanges': len(self.collectors),
            'errors': dict(self.metrics['errors']),
            'recent_opportunities': len(self.arbitrage_opportunities)
        }


# Exemple d'utilisation
async def main():
    """Exemple d'utilisation de l'agrégateur multi-exchanges"""
    
    # Configuration
    config = {
        Exchange.BINANCE: ExchangeConfig(
            enabled=True,
            testnet=True,
            fees={'taker': 0.001, 'maker': 0.001}
        ),
        Exchange.INTERACTIVE_BROKERS: ExchangeConfig(
            enabled=True,
            testnet=True,
            fees={'taker': 0.0005, 'maker': 0.0005}
        )
    }
    
    # Créer l'agrégateur
    aggregator = MultiExchangeAggregator(config)
    
    try:
        # Initialiser
        await aggregator.initialize()
        
        # Callback pour les données unifiées
        async def handle_unified_data(data: UnifiedMarketData):
            logger.info(f"[{data.exchange.value}] {data.symbol}: "
                       f"Bid: {data.bid}, Ask: {data.ask}")
        
        # Callback pour l'arbitrage
        async def handle_arbitrage(opportunity: ArbitrageOpportunity):
            logger.info(f"💰 ARBITRAGE: {opportunity.symbol} "
                       f"Buy on {opportunity.buy_exchange.value} @ {opportunity.buy_price} "
                       f"Sell on {opportunity.sell_exchange.value} @ {opportunity.sell_price} "
                       f"Profit: {opportunity.profit_percentage:.2f}%")
        
        aggregator.add_arbitrage_callback(handle_arbitrage)
        
        # Souscrire aux données
        await aggregator.subscribe_ticker('BTC/USD', handle_unified_data)
        await aggregator.subscribe_orderbook('ETH/USD', depth=10)
        
        # Attendre et afficher les métriques
        await asyncio.sleep(60)
        
        # Afficher le carnet d'ordres consolidé
        consolidated = await aggregator.get_consolidated_orderbook('BTC/USD')
        logger.info(f"Carnet d'ordres consolidé: {consolidated}")
        
        # Afficher les métriques
        metrics = aggregator.get_metrics()
        logger.info(f"Métriques: {metrics}")
        
    finally:
        await aggregator.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())