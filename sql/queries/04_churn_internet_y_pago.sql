-- Consulta 04: Churn por tipo de internet y método de pago
-- Pregunta: ¿Qué combinaciones de internet y forma de pago tienen más bajas (con suficientes clientes para ser confiables)?
-- Qué hace: une las 3 tablas, quita a quienes no tienen internet con WHERE (filtra filas antes de agrupar) y descarta grupos de menos de 100 clientes con HAVING (filtra grupos después de agrupar).
-- Cómo leerla: arriba quedan los segmentos más riesgosos; en el CSV real el pago con cheque electrónico y fibra óptica encabeza la lista.
-- Técnicas: JOIN de 3 tablas, WHERE vs HAVING, LIMIT.
SELECT
    s.internet_service,
    f.payment_method,
    COUNT(*)                        AS clientes,
    SUM(c.churn)                    AS bajas,
    ROUND(100.0 * AVG(c.churn), 2)  AS pct_churn
FROM clientes AS c
INNER JOIN facturacion AS f ON f.customer_id = c.customer_id
INNER JOIN servicios   AS s ON s.customer_id = c.customer_id
WHERE s.internet_service <> 'No'
GROUP BY s.internet_service, f.payment_method
HAVING COUNT(*) >= 100
ORDER BY pct_churn DESC
LIMIT 10;
