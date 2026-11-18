-- Consulta 12: Churn por cuartil de cargo mensual
-- Pregunta: ¿El abandono sube a medida que el cliente paga más cada mes?
-- Qué hace: NTILE(4) divide a los clientes ordenados por cargo mensual en 4 grupos del mismo tamaño (cuartiles) y después se calcula el rango de cargos y la tasa de baja de cada cuartil.
-- Cómo leerla: el cuartil 1 es el de menor cargo y el 4 el de mayor; compara pct_churn entre cuartiles para ver si la relación crece o se rompe.
-- Técnicas: función de ventana NTILE (con customer_id como desempate para que el resultado sea siempre el mismo), CTE, INNER JOIN, GROUP BY.
WITH cuartiles AS (
    SELECT
        c.churn,
        f.monthly_charges,
        NTILE(4) OVER (ORDER BY f.monthly_charges, c.customer_id) AS cuartil_cargo
    FROM clientes AS c
    INNER JOIN facturacion AS f ON f.customer_id = c.customer_id
)
SELECT
    cuartil_cargo,
    COUNT(*)                         AS clientes,
    ROUND(MIN(monthly_charges), 2)   AS cargo_minimo,
    ROUND(MAX(monthly_charges), 2)   AS cargo_maximo,
    ROUND(100.0 * AVG(churn), 2)     AS pct_churn
FROM cuartiles
GROUP BY cuartil_cargo
ORDER BY cuartil_cargo;
