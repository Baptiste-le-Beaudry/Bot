"""
Décorateurs Avancés pour Robot de Trading Algorithmique IA
==========================================================

Ce module implémente des décorateurs sophistiqués pour la résilience, performance
et observabilité des systèmes de trading haute fréquence. Patterns optimisés
pour latence sub-milliseconde et fiabilité de niveau production.

Architecture:
- Circuit breakers avec état adaptatif
- Retry patterns avec backoff exponentiel intelligent
- Rate limiting avec token bucket et sliding window
- Performance monitoring avec métriques temps réel
- Timeout handling avec graceful degradation
- Caching intelligent avec TTL adaptatif
- Thread safety et async/sync compatibility
- Error handling avec classification automatique

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import functools
import hashlib
import inspect
import random
import threading
import time
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union,
    Tuple, Set, Type, Generic, Protocol
)
from datetime import datetime, timedelta
import concurrent.futures

# Pour monitoring des performances
import psutil
import gc

# Import du logger (éviter import circulaire)
from utils.logger import get_structured_logger


# Types génériques
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')


class CircuitState(Enum):
    """États du circuit breaker"""
    CLOSED = "closed"        # Fonctionnement normal
    OPEN = "open"           # Circuit ouvert (échecs)
    HALF_OPEN = "half_open" # Test de récupération


class RetryStrategy(Enum):
    """Stratégies de retry"""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    JITTER = "jitter"


class ErrorType(Enum):
    """Classification des erreurs pour handling intelligent"""
    TRANSIENT = "transient"      # Erreur temporaire (retry)
    PERMANENT = "permanent"      # Erreur permanente (no retry)
    RATE_LIMIT = "rate_limit"    # Rate limiting (backoff)
    TIMEOUT = "timeout"          # Timeout (retry with longer timeout)
    NETWORK = "network"          # Erreur réseau (retry)
    MARKET = "market"           # Erreur de marché (graceful handling)


@dataclass
class CircuitBreakerState:
    """État d'un circuit breaker"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    next_attempt_time: float = 0
    total_requests: int = 0
    total_failures: int = 0


@dataclass
class RetryState:
    """État d'une tentative de retry"""
    attempt: int = 0
    total_attempts: int = 0
    start_time: float = field(default_factory=time.time)
    last_exception: Optional[Exception] = None
    backoff_time: float = 0


@dataclass
class RateLimitState:
    """État du rate limiter"""
    tokens: float = 0
    last_refill: float = field(default_factory=time.time)
    request_times: deque = field(default_factory=lambda: deque(maxlen=1000))


class PerformanceMetrics:
    """Métriques de performance pour décorateurs"""
    
    def __init__(self):
        self.call_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.error_count = 0
        self.last_call_time = 0.0
        self.moving_average = 0.0
        self.percentiles = deque(maxlen=1000)
    
    def record_call(self, duration: float, success: bool = True):
        """Enregistre une métrique d'appel"""
        self.call_count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.last_call_time = duration
        self.percentiles.append(duration)
        
        if not success:
            self.error_count += 1
        
        # Moving average avec fenêtre de 100 appels
        alpha = min(0.1, 2.0 / (self.call_count + 1))
        self.moving_average = alpha * duration + (1 - alpha) * self.moving_average
    
    @property
    def average_time(self) -> float:
        return self.total_time / max(1, self.call_count)
    
    @property
    def error_rate(self) -> float:
        return self.error_count / max(1, self.call_count)
    
    @property
    def success_rate(self) -> float:
        return 1.0 - self.error_rate
    
    def get_percentile(self, p: float) -> float:
        """Calcule le percentile p (0-100)"""
        if not self.percentiles:
            return 0.0
        sorted_times = sorted(self.percentiles)
        index = int(len(sorted_times) * p / 100)
        return sorted_times[min(index, len(sorted_times) - 1)]


# Stockage global des états (thread-safe)
_circuit_states: Dict[str, CircuitBreakerState] = {}
_performance_metrics: Dict[str, PerformanceMetrics] = {}
_rate_limit_states: Dict[str, RateLimitState] = {}
_cache_storage: Dict[str, Tuple[Any, float, float]] = {}  # key -> (value, timestamp, ttl)
_lock = threading.RLock()


def _get_function_key(func: Callable, *args, **kwargs) -> str:
    """Génère une clé unique pour une fonction et ses paramètres"""
    func_name = f"{func.__module__}.{func.__qualname__}"
    
    # Hash des arguments pour cache key
    if args or kwargs:
        arg_str = str(args) + str(sorted(kwargs.items()))
        arg_hash = hashlib.md5(arg_str.encode()).hexdigest()[:8]
        return f"{func_name}:{arg_hash}"
    
    return func_name


def _classify_error(exception: Exception) -> ErrorType:
    """Classifie automatiquement une erreur"""
    error_name = type(exception).__name__.lower()
    error_msg = str(exception).lower()
    
    # Erreurs réseau/connexion (transient)
    if any(keyword in error_name for keyword in ['connection', 'timeout', 'network']):
        return ErrorType.NETWORK
    
    # Rate limiting
    if any(keyword in error_msg for keyword in ['rate limit', 'too many requests', '429']):
        return ErrorType.RATE_LIMIT
    
    # Timeout
    if 'timeout' in error_name or 'timeout' in error_msg:
        return ErrorType.TIMEOUT
    
    # Erreurs de marché (spécifiques trading)
    if any(keyword in error_msg for keyword in ['insufficient', 'balance', 'market closed']):
        return ErrorType.MARKET
    
    # Erreurs permanentes
    if any(keyword in error_name for keyword in ['permission', 'authorization', 'authentication']):
        return ErrorType.PERMANENT
    
    # Par défaut : transient (retry possible)
    return ErrorType.TRANSIENT


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception,
    fallback_function: Optional[Callable] = None,
    monitor_performance: bool = True
):
    """
    Circuit breaker pattern pour prévenir les cascades d'échecs
    
    Args:
        failure_threshold: Nombre d'échecs avant ouverture du circuit
        recovery_timeout: Temps d'attente avant test de récupération (secondes)
        expected_exception: Type d'exception à monitorer
        fallback_function: Fonction de fallback en cas de circuit ouvert
        monitor_performance: Active le monitoring des performances
    """
    def decorator(func: F) -> F:
        func_key = _get_function_key(func)
        logger = get_structured_logger(f"circuit_breaker.{func.__name__}")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _circuit_breaker_sync(
                func, func_key, logger, failure_threshold, recovery_timeout,
                expected_exception, fallback_function, monitor_performance,
                args, kwargs
            )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _circuit_breaker_async(
                func, func_key, logger, failure_threshold, recovery_timeout,
                expected_exception, fallback_function, monitor_performance,
                args, kwargs
            )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


def _circuit_breaker_sync(func, func_key, logger, failure_threshold, recovery_timeout,
                         expected_exception, fallback_function, monitor_performance, args, kwargs):
    """Implémentation synchrone du circuit breaker"""
    with _lock:
        state = _circuit_states.setdefault(func_key, CircuitBreakerState())
        metrics = _performance_metrics.setdefault(func_key, PerformanceMetrics()) if monitor_performance else None
    
    current_time = time.time()
    
    # Vérifier l'état du circuit
    if state.state == CircuitState.OPEN:
        if current_time < state.next_attempt_time:
            logger.warning("circuit_breaker_open", 
                         failure_count=state.failure_count,
                         time_until_retry=state.next_attempt_time - current_time)
            
            if fallback_function:
                return fallback_function(*args, **kwargs)
            else:
                raise Exception(f"Circuit breaker open for {func.__name__}")
        else:
            # Transition vers HALF_OPEN
            with _lock:
                state.state = CircuitState.HALF_OPEN
            logger.info("circuit_breaker_half_open", func_name=func.__name__)
    
    # Exécuter la fonction
    start_time = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        
        # Succès
        with _lock:
            state.success_count += 1
            state.total_requests += 1
            state.last_success_time = current_time
            
            if state.state == CircuitState.HALF_OPEN:
                state.state = CircuitState.CLOSED
                state.failure_count = 0
                logger.info("circuit_breaker_closed", func_name=func.__name__)
        
        if metrics:
            metrics.record_call(execution_time, success=True)
        
        return result
        
    except expected_exception as e:
        execution_time = time.perf_counter() - start_time
        error_type = _classify_error(e)
        
        with _lock:
            state.failure_count += 1
            state.total_requests += 1
            state.total_failures += 1
            state.last_failure_time = current_time
            
            # Ouvrir le circuit si seuil atteint
            if state.failure_count >= failure_threshold:
                state.state = CircuitState.OPEN
                state.next_attempt_time = current_time + recovery_timeout
                logger.error("circuit_breaker_opened", 
                           func_name=func.__name__,
                           failure_count=state.failure_count,
                           error_type=error_type.value)
        
        if metrics:
            metrics.record_call(execution_time, success=False)
        
        raise


async def _circuit_breaker_async(func, func_key, logger, failure_threshold, recovery_timeout,
                                expected_exception, fallback_function, monitor_performance, args, kwargs):
    """Implémentation asynchrone du circuit breaker"""
    # Même logique que sync mais avec await
    with _lock:
        state = _circuit_states.setdefault(func_key, CircuitBreakerState())
        metrics = _performance_metrics.setdefault(func_key, PerformanceMetrics()) if monitor_performance else None
    
    current_time = time.time()
    
    if state.state == CircuitState.OPEN:
        if current_time < state.next_attempt_time:
            logger.warning("circuit_breaker_open", 
                         failure_count=state.failure_count,
                         time_until_retry=state.next_attempt_time - current_time)
            
            if fallback_function:
                if asyncio.iscoroutinefunction(fallback_function):
                    return await fallback_function(*args, **kwargs)
                else:
                    return fallback_function(*args, **kwargs)
            else:
                raise Exception(f"Circuit breaker open for {func.__name__}")
        else:
            with _lock:
                state.state = CircuitState.HALF_OPEN
            logger.info("circuit_breaker_half_open", func_name=func.__name__)
    
    start_time = time.perf_counter()
    try:
        result = await func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        
        with _lock:
            state.success_count += 1
            state.total_requests += 1
            state.last_success_time = current_time
            
            if state.state == CircuitState.HALF_OPEN:
                state.state = CircuitState.CLOSED
                state.failure_count = 0
                logger.info("circuit_breaker_closed", func_name=func.__name__)
        
        if metrics:
            metrics.record_call(execution_time, success=True)
        
        return result
        
    except expected_exception as e:
        execution_time = time.perf_counter() - start_time
        error_type = _classify_error(e)
        
        with _lock:
            state.failure_count += 1
            state.total_requests += 1
            state.total_failures += 1
            state.last_failure_time = current_time
            
            if state.failure_count >= failure_threshold:
                state.state = CircuitState.OPEN
                state.next_attempt_time = current_time + recovery_timeout
                logger.error("circuit_breaker_opened", 
                           func_name=func.__name__,
                           failure_count=state.failure_count,
                           error_type=error_type.value)
        
        if metrics:
            metrics.record_call(execution_time, success=False)
        
        raise


def retry_async(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    jitter: bool = True,
    timeout_per_attempt: Optional[float] = None
):
    """
    Retry automatique avec backoff intelligent pour fonctions async/sync
    
    Args:
        max_attempts: Nombre maximum de tentatives
        backoff_factor: Facteur de multiplication pour backoff exponentiel
        max_backoff: Temps maximum entre tentatives (secondes)
        strategy: Stratégie de backoff
        exceptions: Types d'exceptions à retry
        jitter: Ajoute du random pour éviter thundering herd
        timeout_per_attempt: Timeout par tentative
    """
    def decorator(func: F) -> F:
        func_key = _get_function_key(func)
        logger = get_structured_logger(f"retry.{func.__name__}")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _retry_sync(
                func, func_key, logger, max_attempts, backoff_factor, max_backoff,
                strategy, exceptions, jitter, timeout_per_attempt, args, kwargs
            )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _retry_async(
                func, func_key, logger, max_attempts, backoff_factor, max_backoff,
                strategy, exceptions, jitter, timeout_per_attempt, args, kwargs
            )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


def _calculate_backoff(attempt: int, strategy: RetryStrategy, backoff_factor: float, 
                      max_backoff: float, jitter: bool) -> float:
    """Calcule le temps de backoff selon la stratégie"""
    if strategy == RetryStrategy.FIXED:
        backoff = backoff_factor
    elif strategy == RetryStrategy.LINEAR:
        backoff = attempt * backoff_factor
    elif strategy == RetryStrategy.EXPONENTIAL:
        backoff = backoff_factor * (2 ** (attempt - 1))
    elif strategy == RetryStrategy.FIBONACCI:
        fib = _fibonacci(attempt)
        backoff = backoff_factor * fib
    elif strategy == RetryStrategy.JITTER:
        backoff = random.uniform(0, backoff_factor * (2 ** attempt))
    else:
        backoff = backoff_factor
    
    # Limiter le backoff maximum
    backoff = min(backoff, max_backoff)
    
    # Ajouter du jitter pour éviter thundering herd
    if jitter and strategy != RetryStrategy.JITTER:
        jitter_amount = backoff * 0.1 * random.random()
        backoff += jitter_amount
    
    return backoff


def _fibonacci(n: int) -> int:
    """Calcule le nième nombre de Fibonacci (memoized)"""
    if not hasattr(_fibonacci, 'cache'):
        _fibonacci.cache = {0: 0, 1: 1}
    
    if n in _fibonacci.cache:
        return _fibonacci.cache[n]
    
    _fibonacci.cache[n] = _fibonacci(n-1) + _fibonacci(n-2)
    return _fibonacci.cache[n]


def _retry_sync(func, func_key, logger, max_attempts, backoff_factor, max_backoff,
               strategy, exceptions, jitter, timeout_per_attempt, args, kwargs):
    """Implémentation synchrone du retry"""
    retry_state = RetryState(total_attempts=max_attempts)
    
    for attempt in range(1, max_attempts + 1):
        retry_state.attempt = attempt
        
        try:
            if timeout_per_attempt:
                # Utiliser ThreadPoolExecutor pour timeout
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    return future.result(timeout=timeout_per_attempt)
            else:
                return func(*args, **kwargs)
                
        except exceptions as e:
            retry_state.last_exception = e
            error_type = _classify_error(e)
            
            logger.warning("retry_attempt_failed",
                         attempt=attempt,
                         max_attempts=max_attempts,
                         error_type=error_type.value,
                         error=str(e))
            
            # Ne pas retry sur erreurs permanentes
            if error_type == ErrorType.PERMANENT:
                logger.error("retry_stopped_permanent_error", error=str(e))
                raise
            
            # Dernière tentative
            if attempt == max_attempts:
                logger.error("retry_exhausted", 
                           total_attempts=max_attempts,
                           total_time=time.time() - retry_state.start_time)
                raise
            
            # Calculer backoff et attendre
            backoff_time = _calculate_backoff(attempt, strategy, backoff_factor, max_backoff, jitter)
            retry_state.backoff_time = backoff_time
            
            logger.info("retry_backoff", 
                       backoff_seconds=backoff_time,
                       next_attempt=attempt + 1)
            
            time.sleep(backoff_time)


async def _retry_async(func, func_key, logger, max_attempts, backoff_factor, max_backoff,
                      strategy, exceptions, jitter, timeout_per_attempt, args, kwargs):
    """Implémentation asynchrone du retry"""
    retry_state = RetryState(total_attempts=max_attempts)
    
    for attempt in range(1, max_attempts + 1):
        retry_state.attempt = attempt
        
        try:
            if timeout_per_attempt:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_per_attempt)
            else:
                return await func(*args, **kwargs)
                
        except exceptions as e:
            retry_state.last_exception = e
            error_type = _classify_error(e)
            
            logger.warning("retry_attempt_failed",
                         attempt=attempt,
                         max_attempts=max_attempts,
                         error_type=error_type.value,
                         error=str(e))
            
            if error_type == ErrorType.PERMANENT:
                logger.error("retry_stopped_permanent_error", error=str(e))
                raise
            
            if attempt == max_attempts:
                logger.error("retry_exhausted", 
                           total_attempts=max_attempts,
                           total_time=time.time() - retry_state.start_time)
                raise
            
            backoff_time = _calculate_backoff(attempt, strategy, backoff_factor, max_backoff, jitter)
            retry_state.backoff_time = backoff_time
            
            logger.info("retry_backoff", 
                       backoff_seconds=backoff_time,
                       next_attempt=attempt + 1)
            
            await asyncio.sleep(backoff_time)


def rate_limit(
    calls_per_second: float = 10.0,
    burst_size: Optional[int] = None,
    algorithm: str = "token_bucket",  # ou "sliding_window"
    block_on_limit: bool = True,
    fallback_function: Optional[Callable] = None
):
    """
    Rate limiting avec token bucket ou sliding window
    
    Args:
        calls_per_second: Limite de taux (appels par seconde)
        burst_size: Taille du burst (None = calls_per_second)
        algorithm: Algorithme ("token_bucket" ou "sliding_window")
        block_on_limit: Bloquer ou rejeter quand limite atteinte
        fallback_function: Fonction alternative si limite atteinte
    """
    def decorator(func: F) -> F:
        func_key = _get_function_key(func)
        logger = get_structured_logger(f"rate_limit.{func.__name__}")
        
        if burst_size is None:
            bucket_size = int(calls_per_second)
        else:
            bucket_size = burst_size
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _rate_limit_sync(
                func, func_key, logger, calls_per_second, bucket_size,
                algorithm, block_on_limit, fallback_function, args, kwargs
            )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _rate_limit_async(
                func, func_key, logger, calls_per_second, bucket_size,
                algorithm, block_on_limit, fallback_function, args, kwargs
            )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


def _rate_limit_sync(func, func_key, logger, calls_per_second, bucket_size,
                    algorithm, block_on_limit, fallback_function, args, kwargs):
    """Implémentation synchrone du rate limiting"""
    current_time = time.time()
    
    with _lock:
        state = _rate_limit_states.setdefault(func_key, RateLimitState())
        
        if algorithm == "token_bucket":
            # Remplir le bucket
            time_passed = current_time - state.last_refill
            tokens_to_add = time_passed * calls_per_second
            state.tokens = min(bucket_size, state.tokens + tokens_to_add)
            state.last_refill = current_time
            
            # Vérifier si on peut consommer un token
            if state.tokens >= 1.0:
                state.tokens -= 1.0
                can_proceed = True
            else:
                can_proceed = False
                wait_time = (1.0 - state.tokens) / calls_per_second
        
        elif algorithm == "sliding_window":
            # Nettoyer les anciens appels
            cutoff = current_time - 1.0  # Fenêtre d'1 seconde
            while state.request_times and state.request_times[0] < cutoff:
                state.request_times.popleft()
            
            # Vérifier la limite
            if len(state.request_times) < calls_per_second:
                state.request_times.append(current_time)
                can_proceed = True
            else:
                can_proceed = False
                wait_time = state.request_times[0] + 1.0 - current_time
        
        else:
            raise ValueError(f"Unknown rate limit algorithm: {algorithm}")
    
    if not can_proceed:
        logger.warning("rate_limit_exceeded", 
                      calls_per_second=calls_per_second,
                      wait_time=wait_time)
        
        if fallback_function:
            return fallback_function(*args, **kwargs)
        elif block_on_limit:
            time.sleep(wait_time)
            return func(*args, **kwargs)
        else:
            raise Exception("Rate limit exceeded")
    
    return func(*args, **kwargs)


async def _rate_limit_async(func, func_key, logger, calls_per_second, bucket_size,
                           algorithm, block_on_limit, fallback_function, args, kwargs):
    """Implémentation asynchrone du rate limiting"""
    current_time = time.time()
    
    with _lock:
        state = _rate_limit_states.setdefault(func_key, RateLimitState())
        
        if algorithm == "token_bucket":
            time_passed = current_time - state.last_refill
            tokens_to_add = time_passed * calls_per_second
            state.tokens = min(bucket_size, state.tokens + tokens_to_add)
            state.last_refill = current_time
            
            if state.tokens >= 1.0:
                state.tokens -= 1.0
                can_proceed = True
            else:
                can_proceed = False
                wait_time = (1.0 - state.tokens) / calls_per_second
        
        elif algorithm == "sliding_window":
            cutoff = current_time - 1.0
            while state.request_times and state.request_times[0] < cutoff:
                state.request_times.popleft()
            
            if len(state.request_times) < calls_per_second:
                state.request_times.append(current_time)
                can_proceed = True
            else:
                can_proceed = False
                wait_time = state.request_times[0] + 1.0 - current_time
        
        else:
            raise ValueError(f"Unknown rate limit algorithm: {algorithm}")
    
    if not can_proceed:
        logger.warning("rate_limit_exceeded", 
                      calls_per_second=calls_per_second,
                      wait_time=wait_time)
        
        if fallback_function:
            if asyncio.iscoroutinefunction(fallback_function):
                return await fallback_function(*args, **kwargs)
            else:
                return fallback_function(*args, **kwargs)
        elif block_on_limit:
            await asyncio.sleep(wait_time)
            return await func(*args, **kwargs)
        else:
            raise Exception("Rate limit exceeded")
    
    return await func(*args, **kwargs)


def timeout(
    timeout_seconds: float,
    timeout_exception: Type[Exception] = TimeoutError,
    default_return: Any = None,
    use_default: bool = False
):
    """
    Timeout avec gestion gracieuse pour fonctions sync/async
    
    Args:
        timeout_seconds: Timeout en secondes
        timeout_exception: Exception à lever si timeout
        default_return: Valeur par défaut si timeout
        use_default: Utiliser valeur par défaut au lieu d'exception
    """
    def decorator(func: F) -> F:
        logger = get_structured_logger(f"timeout.{func.__name__}")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                logger.warning("function_timeout", 
                             timeout_seconds=timeout_seconds,
                             function=func.__name__)
                if use_default:
                    return default_return
                else:
                    raise timeout_exception(f"Function {func.__name__} timed out after {timeout_seconds}s")
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.warning("function_timeout", 
                             timeout_seconds=timeout_seconds,
                             function=func.__name__)
                if use_default:
                    return default_return
                else:
                    raise timeout_exception(f"Function {func.__name__} timed out after {timeout_seconds}s")
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


def cache(
    ttl_seconds: float = 300.0,
    max_size: int = 1000,
    cache_key_func: Optional[Callable] = None,
    ignore_kwargs: Optional[Set[str]] = None
):
    """
    Cache intelligent avec TTL et LRU
    
    Args:
        ttl_seconds: Time to live en secondes
        max_size: Taille maximum du cache
        cache_key_func: Fonction personnalisée pour générer les clés
        ignore_kwargs: Arguments à ignorer pour la clé de cache
    """
    def decorator(func: F) -> F:
        logger = get_structured_logger(f"cache.{func.__name__}")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Générer la clé de cache
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                filtered_kwargs = kwargs.copy()
                if ignore_kwargs:
                    for key in ignore_kwargs:
                        filtered_kwargs.pop(key, None)
                cache_key = _get_function_key(func, *args, **filtered_kwargs)
            
            current_time = time.time()
            
            # Vérifier le cache
            with _lock:
                if cache_key in _cache_storage:
                    value, timestamp, ttl = _cache_storage[cache_key]
                    if current_time - timestamp < ttl:
                        logger.debug("cache_hit", cache_key=cache_key)
                        return value
                    else:
                        # Expired
                        del _cache_storage[cache_key]
                        logger.debug("cache_expired", cache_key=cache_key)
            
            # Cache miss - exécuter la fonction
            logger.debug("cache_miss", cache_key=cache_key)
            result = func(*args, **kwargs)
            
            # Stocker dans le cache
            with _lock:
                # Nettoyer le cache si plein
                if len(_cache_storage) >= max_size:
                    # Supprimer les plus anciens
                    oldest_keys = sorted(_cache_storage.keys(), 
                                       key=lambda k: _cache_storage[k][1])[:max_size//4]
                    for old_key in oldest_keys:
                        del _cache_storage[old_key]
                
                _cache_storage[cache_key] = (result, current_time, ttl_seconds)
            
            return result
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Même logique mais avec await
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                filtered_kwargs = kwargs.copy()
                if ignore_kwargs:
                    for key in ignore_kwargs:
                        filtered_kwargs.pop(key, None)
                cache_key = _get_function_key(func, *args, **filtered_kwargs)
            
            current_time = time.time()
            
            with _lock:
                if cache_key in _cache_storage:
                    value, timestamp, ttl = _cache_storage[cache_key]
                    if current_time - timestamp < ttl:
                        logger.debug("cache_hit", cache_key=cache_key)
                        return value
                    else:
                        del _cache_storage[cache_key]
                        logger.debug("cache_expired", cache_key=cache_key)
            
            logger.debug("cache_miss", cache_key=cache_key)
            result = await func(*args, **kwargs)
            
            with _lock:
                if len(_cache_storage) >= max_size:
                    oldest_keys = sorted(_cache_storage.keys(), 
                                       key=lambda k: _cache_storage[k][1])[:max_size//4]
                    for old_key in oldest_keys:
                        del _cache_storage[old_key]
                
                _cache_storage[cache_key] = (result, current_time, ttl_seconds)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


def measure_performance(
    track_memory: bool = True,
    track_cpu: bool = True,
    percentiles: List[int] = [50, 95, 99]
):
    """
    Mesure automatique des performances avec métriques détaillées
    
    Args:
        track_memory: Suivre l'usage mémoire
        track_cpu: Suivre l'usage CPU
        percentiles: Percentiles à calculer
    """
    def decorator(func: F) -> F:
        func_key = _get_function_key(func)
        logger = get_structured_logger(f"perf.{func.__name__}")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _measure_performance_sync(
                func, func_key, logger, track_memory, track_cpu, percentiles, args, kwargs
            )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _measure_performance_async(
                func, func_key, logger, track_memory, track_cpu, percentiles, args, kwargs
            )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


def _measure_performance_sync(func, func_key, logger, track_memory, track_cpu, percentiles, args, kwargs):
    """Mesure de performance synchrone"""
    process = psutil.Process() if track_memory or track_cpu else None
    start_cpu = process.cpu_percent() if track_cpu else 0
    start_memory = process.memory_info().rss if track_memory else 0
    start_time = time.perf_counter()
    
    try:
        result = func(*args, **kwargs)
        success = True
    except Exception as e:
        result = e
        success = False
        raise
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Métriques système
        metrics_data = {
            "duration_ms": duration * 1000,
            "success": success
        }
        
        if track_cpu and process:
            metrics_data["cpu_percent"] = process.cpu_percent()
        
        if track_memory and process:
            end_memory = process.memory_info().rss
            metrics_data["memory_delta_mb"] = (end_memory - start_memory) / 1024 / 1024
            metrics_data["memory_usage_mb"] = end_memory / 1024 / 1024
        
        # Enregistrer dans les métriques globales
        with _lock:
            metrics = _performance_metrics.setdefault(func_key, PerformanceMetrics())
            metrics.record_call(duration, success)
            
            # Ajouter les percentiles
            for p in percentiles:
                metrics_data[f"p{p}_ms"] = metrics.get_percentile(p) * 1000
        
        logger.info("performance_measured", **metrics_data)
    
    return result


async def _measure_performance_async(func, func_key, logger, track_memory, track_cpu, percentiles, args, kwargs):
    """Mesure de performance asynchrone"""
    process = psutil.Process() if track_memory or track_cpu else None
    start_cpu = process.cpu_percent() if track_cpu else 0
    start_memory = process.memory_info().rss if track_memory else 0
    start_time = time.perf_counter()
    
    try:
        result = await func(*args, **kwargs)
        success = True
    except Exception as e:
        result = e
        success = False
        raise
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        metrics_data = {
            "duration_ms": duration * 1000,
            "success": success
        }
        
        if track_cpu and process:
            metrics_data["cpu_percent"] = process.cpu_percent()
        
        if track_memory and process:
            end_memory = process.memory_info().rss
            metrics_data["memory_delta_mb"] = (end_memory - start_memory) / 1024 / 1024
            metrics_data["memory_usage_mb"] = end_memory / 1024 / 1024
        
        with _lock:
            metrics = _performance_metrics.setdefault(func_key, PerformanceMetrics())
            metrics.record_call(duration, success)
            
            for p in percentiles:
                metrics_data[f"p{p}_ms"] = metrics.get_percentile(p) * 1000
        
        logger.info("performance_measured", **metrics_data)
    
    return result


def thread_safe(lock: Optional[threading.Lock] = None):
    """
    Garantit la thread safety d'une fonction
    
    Args:
        lock: Lock personnalisé (None = utilise un lock global)
    """
    def decorator(func: F) -> F:
        func_lock = lock or threading.RLock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with func_lock:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def fallback(fallback_function: Callable, exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    Fournit une fonction de fallback en cas d'erreur
    
    Args:
        fallback_function: Fonction à exécuter en cas d'erreur
        exceptions: Types d'exceptions qui déclenchent le fallback
    """
    def decorator(func: F) -> F:
        logger = get_structured_logger(f"fallback.{func.__name__}")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.warning("fallback_triggered", 
                             error=str(e),
                             fallback_function=fallback_function.__name__)
                return fallback_function(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                logger.warning("fallback_triggered", 
                             error=str(e),
                             fallback_function=fallback_function.__name__)
                if asyncio.iscoroutinefunction(fallback_function):
                    return await fallback_function(*args, **kwargs)
                else:
                    return fallback_function(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


# Fonctions utilitaires pour monitoring
def get_circuit_breaker_stats(func_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Retourne les statistiques des circuit breakers"""
    with _lock:
        if func_name:
            key = next((k for k in _circuit_states.keys() if func_name in k), None)
            if key:
                state = _circuit_states[key]
                return {key: {
                    "state": state.state.value,
                    "failure_count": state.failure_count,
                    "success_count": state.success_count,
                    "total_requests": state.total_requests,
                    "failure_rate": state.total_failures / max(1, state.total_requests),
                    "last_failure": state.last_failure_time,
                    "next_attempt": state.next_attempt_time
                }}
            return {}
        
        return {
            key: {
                "state": state.state.value,
                "failure_count": state.failure_count,
                "success_count": state.success_count,
                "total_requests": state.total_requests,
                "failure_rate": state.total_failures / max(1, state.total_requests),
                "last_failure": state.last_failure_time,
                "next_attempt": state.next_attempt_time
            }
            for key, state in _circuit_states.items()
        }


def get_performance_stats(func_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Retourne les statistiques de performance"""
    with _lock:
        if func_name:
            key = next((k for k in _performance_metrics.keys() if func_name in k), None)
            if key:
                metrics = _performance_metrics[key]
                return {key: {
                    "call_count": metrics.call_count,
                    "average_time_ms": metrics.average_time * 1000,
                    "min_time_ms": metrics.min_time * 1000 if metrics.min_time != float('inf') else 0,
                    "max_time_ms": metrics.max_time * 1000,
                    "moving_average_ms": metrics.moving_average * 1000,
                    "error_rate": metrics.error_rate,
                    "success_rate": metrics.success_rate,
                    "p50_ms": metrics.get_percentile(50) * 1000,
                    "p95_ms": metrics.get_percentile(95) * 1000,
                    "p99_ms": metrics.get_percentile(99) * 1000
                }}
            return {}
        
        return {
            key: {
                "call_count": metrics.call_count,
                "average_time_ms": metrics.average_time * 1000,
                "min_time_ms": metrics.min_time * 1000 if metrics.min_time != float('inf') else 0,
                "max_time_ms": metrics.max_time * 1000,
                "moving_average_ms": metrics.moving_average * 1000,
                "error_rate": metrics.error_rate,
                "success_rate": metrics.success_rate,
                "p50_ms": metrics.get_percentile(50) * 1000,
                "p95_ms": metrics.get_percentile(95) * 1000,
                "p99_ms": metrics.get_percentile(99) * 1000
            }
            for key, metrics in _performance_metrics.items()
        }


def clear_stats():
    """Nettoie toutes les statistiques"""
    with _lock:
        _circuit_states.clear()
        _performance_metrics.clear()
        _rate_limit_states.clear()
        _cache_storage.clear()


# Décorateur combiné pour usage facile
def resilient(
    max_retries: int = 3,
    circuit_threshold: int = 5,
    timeout_seconds: float = 30.0,
    rate_limit_calls: float = 10.0,
    cache_ttl: float = 60.0,
    enable_performance_tracking: bool = True
):
    """
    Décorateur combiné avec tous les patterns de résilience
    
    Args:
        max_retries: Nombre de retries
        circuit_threshold: Seuil du circuit breaker
        timeout_seconds: Timeout par appel
        rate_limit_calls: Limite de taux (appels/sec)
        cache_ttl: TTL du cache (secondes)
        enable_performance_tracking: Active le tracking des performances
    """
    def decorator(func: F) -> F:
        # Appliquer tous les décorateurs
        decorated_func = func
        
        if enable_performance_tracking:
            decorated_func = measure_performance()(decorated_func)
        
        if cache_ttl > 0:
            decorated_func = cache(ttl_seconds=cache_ttl)(decorated_func)
        
        if rate_limit_calls > 0:
            decorated_func = rate_limit(calls_per_second=rate_limit_calls)(decorated_func)
        
        if timeout_seconds > 0:
            decorated_func = timeout(timeout_seconds=timeout_seconds)(decorated_func)
        
        if max_retries > 0:
            decorated_func = retry_async(max_attempts=max_retries)(decorated_func)
        
        if circuit_threshold > 0:
            decorated_func = circuit_breaker(failure_threshold=circuit_threshold)(decorated_func)
        
        return decorated_func
    
    return decorator


# Exports principaux
__all__ = [
    'circuit_breaker',
    'retry_async', 
    'rate_limit',
    'timeout',
    'cache',
    'measure_performance',
    'thread_safe',
    'fallback',
    'resilient',
    'get_circuit_breaker_stats',
    'get_performance_stats',
    'clear_stats',
    'CircuitState',
    'RetryStrategy',
    'ErrorType'
]


if __name__ == "__main__":
    # Tests des décorateurs
    import asyncio
    import random
    
    async def test_decorators():
        print("🚀 Testing Trading Decorators System...")
        
        # Test circuit breaker
        @circuit_breaker(failure_threshold=3, recovery_timeout=2.0)
        async def flaky_function(should_fail: bool = False):
            if should_fail:
                raise Exception("Simulated failure")
            return "success"
        
        # Test retry avec backoff
        @retry_async(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL)
        async def retry_function(fail_count: int = 2):
            retry_function.attempts = getattr(retry_function, 'attempts', 0) + 1
            if retry_function.attempts <= fail_count:
                raise Exception(f"Attempt {retry_function.attempts} failed")
            return f"Success after {retry_function.attempts} attempts"
        
        # Test rate limiting
        @rate_limit(calls_per_second=2.0)
        async def rate_limited_function():
            return f"Called at {time.time()}"
        
        # Test cache
        @cache(ttl_seconds=2.0)
        async def cached_function(value: int):
            await asyncio.sleep(0.1)  # Simulate expensive operation
            return value * 2
        
        # Test performance measurement
        @measure_performance(track_memory=True, track_cpu=True)
        async def performance_function():
            # Simulate some work
            await asyncio.sleep(0.01)
            return random.randint(1, 100)
        
        # Test combiné
        @resilient(
            max_retries=2,
            circuit_threshold=3,
            timeout_seconds=5.0,
            rate_limit_calls=5.0,
            cache_ttl=1.0
        )
        async def resilient_function(data: str):
            await asyncio.sleep(0.01)
            return f"processed: {data}"
        
        print("📊 Testing individual decorators...")
        
        # Test des succès
        result = await flaky_function(should_fail=False)
        print(f"✅ Circuit breaker success: {result}")
        
        # Test retry
        retry_function.attempts = 0
        result = await retry_function(fail_count=1)
        print(f"✅ Retry success: {result}")
        
        # Test rate limiting
        start_time = time.time()
        for i in range(3):
            result = await rate_limited_function()
            print(f"✅ Rate limited call {i}: {time.time() - start_time:.2f}s")
        
        # Test cache (premier appel)
        start_time = time.time()
        result1 = await cached_function(42)
        duration1 = time.time() - start_time
        
        # Test cache (deuxième appel - devrait être en cache)
        start_time = time.time()
        result2 = await cached_function(42)
        duration2 = time.time() - start_time
        
        print(f"✅ Cache test: first={duration1:.3f}s, cached={duration2:.3f}s")
        
        # Test performance
        result = await performance_function()
        print(f"✅ Performance measured: {result}")
        
        # Test fonction résiliente
        result = await resilient_function("test_data")
        print(f"✅ Resilient function: {result}")
        
        print("\n📈 Performance Statistics:")
        perf_stats = get_performance_stats()
        for func_name, stats in perf_stats.items():
            if 'performance_function' in func_name:
                print(f"  {func_name}: {stats['call_count']} calls, "
                      f"avg={stats['average_time_ms']:.2f}ms, "
                      f"p95={stats['p95_ms']:.2f}ms")
        
        print("\n🔌 Circuit Breaker Statistics:")
        cb_stats = get_circuit_breaker_stats()
        for func_name, stats in cb_stats.items():
            print(f"  {func_name}: state={stats['state']}, "
                  f"requests={stats['total_requests']}, "
                  f"failures={stats['failure_count']}")
        
        # Test des échecs pour circuit breaker
        print("\n🔥 Testing failure scenarios...")
        try:
            for i in range(5):
                try:
                    await flaky_function(should_fail=True)
                except Exception as e:
                    print(f"  Expected failure {i+1}: {e}")
        except Exception as e:
            print(f"✅ Circuit breaker opened: {e}")
        
        print("\n✅ All decorator tests completed!")
        
        # Clean up
        clear_stats()
    
    # Run tests
    asyncio.run(test_decorators())