-- Consulta 01: Tasa de churn global
-- Pregunta: ¿Qué porcentaje de los clientes se dio de baja?
-- Qué hace: cuenta todos los clientes, suma los que tienen churn = 1 y calcula el porcentaje. Solo usa la tabla clientes.
-- Cómo leerla: con el CSV real debe dar 7043 clientes, 1869 bajas y 26.54 %, igual que el EDA del notebook (Día 1).
-- Técnicas: agregación (COUNT, SUM, ROUND).
SELECT
    COUNT(*)                                  AS clientes,
    SUM(churn)                                AS bajas,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2)   AS pct_churn
FROM clientes;
