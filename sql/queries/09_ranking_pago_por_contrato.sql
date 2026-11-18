-- Consulta 09: Ranking de métodos de pago dentro de cada contrato
-- Pregunta: Dentro de cada tipo de contrato, ¿qué método de pago tiene más churn?
-- Qué hace: una CTE calcula la tasa de baja por contrato y método de pago; después RANK() OVER (PARTITION BY contract ...) numera los métodos de 1 en adelante, reiniciando el conteo en cada contrato.
-- Cómo leerla: el ranking 1 de cada contrato es su método de pago más riesgoso; a diferencia de GROUP BY, la función de ventana no colapsa las filas.
-- Técnicas: CTE, función de ventana RANK con PARTITION BY.
WITH tasas AS (
    SELECT
        f.contract,
        f.payment_method,
        COUNT(*)                        AS clientes,
        ROUND(100.0 * AVG(c.churn), 2)  AS pct_churn
    FROM clientes AS c
    INNER JOIN facturacion AS f ON f.customer_id = c.customer_id
    GROUP BY f.contract, f.payment_method
)
SELECT
    contract,
    payment_method,
    clientes,
    pct_churn,
    RANK() OVER (PARTITION BY contract ORDER BY pct_churn DESC) AS ranking_en_contrato
FROM tasas
ORDER BY contract, ranking_en_contrato;
