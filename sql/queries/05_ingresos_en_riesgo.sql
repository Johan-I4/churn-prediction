-- Consulta 05: Ingreso mensual perdido por tipo de contrato
-- Pregunta: ¿Cuánto dinero al mes representan los clientes que se fueron, y en qué contratos se concentra?
-- Qué hace: suma el cargo mensual de todos los clientes y, con un CASE dentro del SUM, solo el de quienes tienen churn = 1; después calcula el porcentaje perdido por contrato.
-- Cómo leerla: el contrato con mayor pct_ingreso_perdido (en el CSV real, mes a mes, cerca de la mitad de su ingreso) es donde más conviene gastar en retención.
-- Técnicas: INNER JOIN, SUM condicional con CASE, porcentaje entre dos agregados.
SELECT
    f.contract,
    ROUND(SUM(f.monthly_charges), 2)                                            AS ingreso_mensual_total,
    ROUND(SUM(CASE WHEN c.churn = 1 THEN f.monthly_charges ELSE 0 END), 2)      AS ingreso_mensual_de_bajas,
    ROUND(100.0 * SUM(CASE WHEN c.churn = 1 THEN f.monthly_charges ELSE 0 END)
                / SUM(f.monthly_charges), 2)                                    AS pct_ingreso_perdido
FROM clientes AS c
INNER JOIN facturacion AS f ON f.customer_id = c.customer_id
GROUP BY f.contract
ORDER BY pct_ingreso_perdido DESC;
