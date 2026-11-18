-- Consulta 07: Revisión de integridad con LEFT JOIN
-- Pregunta: ¿Hay clientes que no tengan fila en servicios o en facturacion?
-- Qué hace: parte de clientes y hace LEFT JOIN a las otras dos tablas, que conserva todos los clientes aunque no haya pareja; si no la hay, las columnas de la otra tabla salen NULL y se cuentan.
-- Cómo leerla: debe dar 0 y 0. Un INNER JOIN habría ocultado el problema porque descarta en silencio a los clientes sin pareja.
-- Técnicas: LEFT JOIN, IS NULL, suma de condiciones (en SQLite una condición vale 1 o 0).
SELECT
    COUNT(*)                        AS clientes,
    SUM(f.customer_id IS NULL)      AS sin_facturacion,
    SUM(s.customer_id IS NULL)      AS sin_servicios
FROM clientes AS c
LEFT JOIN facturacion AS f ON f.customer_id = c.customer_id
LEFT JOIN servicios   AS s ON s.customer_id = c.customer_id;
