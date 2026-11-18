-- Consulta 11: Segmentos con churn por encima del promedio
-- Pregunta: ¿Qué combinaciones de contrato e internet abandonan más que el promedio general, y cuántas veces más?
-- Qué hace: agrupa por contrato e internet y con HAVING se queda solo con grupos de 50 clientes o más cuya tasa supera a la global, calculada en una subconsulta escalar sobre toda la tabla clientes.
-- Cómo leerla: la columna veces_el_promedio dice cuántas veces supera cada segmento al churn global; 2.0 significa el doble.
-- Técnicas: JOIN de 3 tablas, GROUP BY de dos columnas, HAVING con subconsulta.
SELECT
    f.contract,
    s.internet_service,
    COUNT(*)                                                         AS clientes,
    ROUND(100.0 * AVG(c.churn), 2)                                   AS pct_churn,
    ROUND(AVG(c.churn) / (SELECT AVG(churn) FROM clientes), 2)       AS veces_el_promedio
FROM clientes AS c
INNER JOIN facturacion AS f ON f.customer_id = c.customer_id
INNER JOIN servicios   AS s ON s.customer_id = c.customer_id
GROUP BY f.contract, s.internet_service
HAVING COUNT(*) >= 50
   AND AVG(c.churn) > (SELECT AVG(churn) FROM clientes)
ORDER BY veces_el_promedio DESC;
