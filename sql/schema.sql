-- Esquema relacional del proyecto de churn (SQLite).
-- El CSV original (una fila por cliente) se divide en 3 tablas que comparten
-- la llave customer_id: clientes (perfil + variable objetivo), servicios
-- (que tiene contratado) y facturacion (contrato y cargos).

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS facturacion;
DROP TABLE IF EXISTS servicios;
DROP TABLE IF EXISTS clientes;

CREATE TABLE clientes (
    customer_id    TEXT PRIMARY KEY,
    gender         TEXT    NOT NULL CHECK (gender IN ('Male', 'Female')),
    senior_citizen INTEGER NOT NULL CHECK (senior_citizen IN (0, 1)),
    partner        TEXT    NOT NULL CHECK (partner IN ('Yes', 'No')),
    dependents     TEXT    NOT NULL CHECK (dependents IN ('Yes', 'No')),
    tenure         INTEGER NOT NULL CHECK (tenure >= 0),          -- meses como cliente
    churn          INTEGER NOT NULL CHECK (churn IN (0, 1))       -- 1 = se dio de baja
);

CREATE TABLE servicios (
    customer_id       TEXT PRIMARY KEY REFERENCES clientes (customer_id),
    phone_service     TEXT NOT NULL,
    multiple_lines    TEXT NOT NULL,
    internet_service  TEXT NOT NULL,
    online_security   TEXT NOT NULL,
    online_backup     TEXT NOT NULL,
    device_protection TEXT NOT NULL,
    tech_support      TEXT NOT NULL,
    streaming_tv      TEXT NOT NULL,
    streaming_movies  TEXT NOT NULL
);

CREATE TABLE facturacion (
    customer_id       TEXT PRIMARY KEY REFERENCES clientes (customer_id),
    contract          TEXT NOT NULL,
    paperless_billing TEXT NOT NULL,
    payment_method    TEXT NOT NULL,
    monthly_charges   REAL NOT NULL CHECK (monthly_charges >= 0),
    total_charges     REAL NOT NULL CHECK (total_charges >= 0)    -- vacios del CSV = 0
);

-- Indices sobre las columnas que mas se usan en GROUP BY y filtros
CREATE INDEX idx_facturacion_contract ON facturacion (contract);
CREATE INDEX idx_servicios_internet   ON servicios (internet_service);
CREATE INDEX idx_clientes_tenure      ON clientes (tenure);
