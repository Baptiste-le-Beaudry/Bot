-- Database initialization script for AI Trading Robot
-- Creates TimescaleDB extensions and initial schema

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set default search path
SET search_path TO trading, public;

-- =============================================================================
-- Market Data Tables
-- =============================================================================

-- OHLCV data
CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    quote_volume DECIMAL(20,8),
    trades_count INTEGER,
    taker_buy_volume DECIMAL(20,8),
    taker_buy_quote_volume DECIMAL(20,8),
    PRIMARY KEY (timestamp, exchange, symbol, interval)
);

-- Convert to hypertable
SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_exchange ON market_data (exchange, symbol, timestamp DESC);

-- Tick data
CREATE TABLE IF NOT EXISTS tick_data (
    timestamp TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    trade_id VARCHAR(100),
    PRIMARY KEY (timestamp, exchange, symbol, trade_id)
);

SELECT create_hypertable('tick_data', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_tick_data_symbol ON tick_data (symbol, timestamp DESC);

-- Order book snapshots
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    bids JSONB NOT NULL,
    asks JSONB NOT NULL,
    sequence_id BIGINT,
    checksum VARCHAR(64),
    PRIMARY KEY (timestamp, exchange, symbol)
);

SELECT create_hypertable('orderbook_snapshots', 'timestamp', if_not_exists => TRUE);

-- =============================================================================
-- Trading Tables
-- =============================================================================

-- Trading strategies
CREATE TABLE IF NOT EXISTS strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(50) NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Signals
CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    strategy_id UUID NOT NULL REFERENCES strategies(id),
    symbol VARCHAR(50) NOT NULL,
    signal_type VARCHAR(20) NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'HOLD')),
    strength DECIMAL(5,4) NOT NULL CHECK (strength >= 0 AND strength <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals (strategy_id, timestamp DESC);

-- Orders
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id BIGINT REFERENCES signals(id),
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8),
    status VARCHAR(20) NOT NULL,
    exchange_order_id VARCHAR(100),
    filled_quantity DECIMAL(20,8) DEFAULT 0,
    average_price DECIMAL(20,8),
    commission DECIMAL(20,8) DEFAULT 0,
    commission_asset VARCHAR(20),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders (symbol, created_at DESC);

-- Trades (filled orders)
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    order_id UUID NOT NULL REFERENCES orders(id),
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(4) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    commission DECIMAL(20,8) DEFAULT 0,
    commission_asset VARCHAR(20),
    realized_pnl DECIMAL(20,8),
    timestamp TIMESTAMPTZ NOT NULL,
    exchange_trade_id VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, timestamp DESC);

-- Positions
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID NOT NULL REFERENCES strategies(id),
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(5) NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    quantity DECIMAL(20,8) NOT NULL,
    entry_price DECIMAL(20,8) NOT NULL,
    current_price DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8),
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN',
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    UNIQUE(strategy_id, symbol, status)
);

CREATE INDEX IF NOT EXISTS idx_positions_status ON positions (status);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions (strategy_id, status);

-- =============================================================================
-- Machine Learning Tables
-- =============================================================================

-- ML models
CREATE TABLE IF NOT EXISTS ml.models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    type VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    model_path VARCHAR(500),
    status VARCHAR(20) NOT NULL DEFAULT 'TRAINING',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trained_at TIMESTAMPTZ,
    UNIQUE(name, version)
);

-- Training history
CREATE TABLE IF NOT EXISTS ml.training_history (
    id BIGSERIAL PRIMARY KEY,
    model_id UUID NOT NULL REFERENCES ml.models(id),
    epoch INTEGER NOT NULL,
    train_loss DECIMAL(10,6),
    val_loss DECIMAL(10,6),
    metrics JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Features
CREATE TABLE IF NOT EXISTS ml.features (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    feature_set VARCHAR(100) NOT NULL,
    features JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, feature_set)
);

SELECT create_hypertable('ml.features', 'timestamp', if_not_exists => TRUE);

-- Predictions
CREATE TABLE IF NOT EXISTS ml.predictions (
    id BIGSERIAL PRIMARY KEY,
    model_id UUID NOT NULL REFERENCES ml.models(id),
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL,
    prediction JSONB NOT NULL,
    confidence DECIMAL(5,4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON ml.predictions (timestamp DESC);

-- =============================================================================
-- Risk Management Tables
-- =============================================================================

-- Risk limits
CREATE TABLE IF NOT EXISTS risk_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    limit_type VARCHAR(50) NOT NULL,
    value DECIMAL(20,8) NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Risk events
CREATE TABLE IF NOT EXISTS risk_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    description TEXT,
    metadata JSONB DEFAULT '{}',
    resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_risk_events_timestamp ON risk_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_events_severity ON risk_events (severity, resolved);

-- =============================================================================
-- Performance Analytics Tables
-- =============================================================================

-- Daily performance
CREATE TABLE IF NOT EXISTS daily_performance (
    date DATE NOT NULL,
    strategy_id UUID REFERENCES strategies(id),
    starting_balance DECIMAL(20,8) NOT NULL,
    ending_balance DECIMAL(20,8) NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    gross_profit DECIMAL(20,8) DEFAULT 0,
    gross_loss DECIMAL(20,8) DEFAULT 0,
    net_profit DECIMAL(20,8) DEFAULT 0,
    max_drawdown DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,4),
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(10,4),
    PRIMARY KEY (date, strategy_id)
);

-- Portfolio snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    total_value DECIMAL(20,8) NOT NULL,
    cash_balance DECIMAL(20,8) NOT NULL,
    positions_value DECIMAL(20,8) NOT NULL,
    unrealized_pnl DECIMAL(20,8) DEFAULT 0,
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    positions_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);

SELECT create_hypertable('portfolio_snapshots', 'timestamp', if_not_exists => TRUE);

-- =============================================================================
-- Audit Tables
-- =============================================================================

-- Audit log
CREATE TABLE IF NOT EXISTS audit.logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50),
    entity_id VARCHAR(100),
    changes JSONB,
    ip_address INET,
    user_agent TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit.logs (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON audit.logs (user_id, timestamp DESC);

-- =============================================================================
-- Functions and Triggers
-- =============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update trigger to relevant tables
CREATE TRIGGER update_strategies_updated_at BEFORE UPDATE ON strategies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_risk_limits_updated_at BEFORE UPDATE ON risk_limits
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- Continuous Aggregates (TimescaleDB)
-- =============================================================================

-- 1-minute OHLCV
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1m
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 minute', timestamp) AS bucket,
    exchange,
    symbol,
    first(close, timestamp) AS open,
    max(close) AS high,
    min(close) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume
FROM market_data
WHERE interval = '1m'
GROUP BY bucket, exchange, symbol
WITH NO DATA;

-- 5-minute OHLCV
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_5m
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('5 minutes', timestamp) AS bucket,
    exchange,
    symbol,
    first(close, timestamp) AS open,
    max(close) AS high,
    min(close) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume
FROM market_data
WHERE interval = '1m'
GROUP BY bucket, exchange, symbol
WITH NO DATA;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep tick data for 7 days
SELECT add_retention_policy('tick_data', INTERVAL '7 days', if_not_exists => TRUE);

-- Keep 1-minute data for 30 days
SELECT add_retention_policy('market_data', INTERVAL '30 days', if_not_exists => TRUE);

-- Keep order book snapshots for 1 day
SELECT add_retention_policy('orderbook_snapshots', INTERVAL '1 day', if_not_exists => TRUE);

-- Keep features for 90 days
SELECT add_retention_policy('ml.features', INTERVAL '90 days', if_not_exists => TRUE);

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Compress market data older than 7 days
SELECT add_compression_policy('market_data', INTERVAL '7 days', if_not_exists => TRUE);

-- Compress tick data older than 1 day
SELECT add_compression_policy('tick_data', INTERVAL '1 day', if_not_exists => TRUE);

-- =============================================================================
-- Initial Data
-- =============================================================================

-- Insert default strategies
INSERT INTO strategies (name, type, config, enabled) VALUES
    ('Statistical Arbitrage', 'ARBITRAGE', '{"lookback_period": 100, "entry_z_score": 2.0, "exit_z_score": 0.5}', true),
    ('Market Making', 'MARKET_MAKING', '{"spread": 0.002, "order_levels": 5, "order_amount": 0.1}', true),
    ('Scalping', 'SCALPING', '{"profit_target": 0.001, "stop_loss": 0.0005, "max_hold_time": 300}', true),
    ('ML Ensemble', 'ML_ENSEMBLE', '{"models": ["dqn", "ppo", "sac"], "voting": "weighted"}', false)
ON CONFLICT (name) DO NOTHING;

-- Insert default risk limits
INSERT INTO risk_limits (name, limit_type, value, enabled) VALUES
    ('Max Position Size', 'POSITION_SIZE_PERCENT', 0.1, true),
    ('Max Daily Loss', 'DAILY_LOSS_PERCENT', 0.05, true),
    ('Max Drawdown', 'MAX_DRAWDOWN_PERCENT', 0.2, true),
    ('Max Leverage', 'MAX_LEVERAGE', 3.0, true),
    ('Max Correlation', 'MAX_CORRELATION', 0.8, true)
ON CONFLICT (name) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON SCHEMA ml TO trader;
GRANT ALL PRIVILEGES ON SCHEMA audit TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA audit TO trader;

-- =============================================================================
-- Performance Optimization
-- =============================================================================

-- Update table statistics
ANALYZE;

-- Show configuration
SELECT 'Database initialization complete!' AS status;
SELECT current_database() AS database, current_user AS user, version() AS postgres_version;
SELECT default_version, installed_version FROM pg_available_extensions WHERE name = 'timescaledb';