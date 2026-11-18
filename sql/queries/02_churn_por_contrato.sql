-- Consulta 02: Churn por tipo de contrato
-- Pregunta: ¿Cambia el abandono según el contrato (mes a mes, 1 año, 2 años)?
-- Qué hace: une clientes con facturacion por customer_id (INNER JOIN), agrupa por contrato y promedia la columna churn, que vale 0 o 1.
-- Cómo leerla: con el CSV real debe dar 42.71 % mes a mes, 11.27 % a un año y 2.83 % a dos años, igual que el notebook.
-- Técnicas: INNER JOIN, GROUP BY, AVG sobre una variable 0/1 (el promedio es la tasa).
SELECT
    f.contract,
    COUNT(*)                        AS clientes,
    SUM(c.churn)                    AS bajas,
    ROUND(100.0 * AVG(c.churn), 2)  AS pct_churn
FROM clientes AS c
INNER JOIN facturacion AS f ON f.customer_id = c.customer_id
GROUP BY f.contract
ORDER BY pct_churn DESC;
