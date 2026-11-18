-- Consulta 06: Cargo mensual de quienes se van y de quienes se quedan
-- Pregunta: ¿Pagan más los clientes que cancelan?
-- Qué hace: une clientes con facturacion, separa a los clientes en 'Se quedan' y 'Se van' según churn y calcula promedio, mínimo y máximo del cargo mensual de cada grupo.
-- Cómo leerla: con el CSV real el promedio es 61.27 para quienes se quedan y 74.44 para quienes se van, igual que el notebook.
-- Técnicas: INNER JOIN, CASE para etiquetar, AVG, MIN y MAX.
SELECT
    CASE c.churn WHEN 1 THEN 'Se van' ELSE 'Se quedan' END  AS grupo,
    COUNT(*)                                                 AS clientes,
    ROUND(AVG(f.monthly_charges), 2)                         AS cargo_promedio,
    ROUND(MIN(f.monthly_charges), 2)                         AS cargo_minimo,
    ROUND(MAX(f.monthly_charges), 2)                         AS cargo_maximo
FROM clientes AS c
INNER JOIN facturacion AS f ON f.customer_id = c.customer_id
GROUP BY c.churn
ORDER BY c.churn;
