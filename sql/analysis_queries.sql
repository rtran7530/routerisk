-- Explore route-level variability using SQLite (source table: trips).
-- Routes with at least 20 observations, highest empirical variance first.
SELECT pickup_zone, dropoff_zone, COUNT(*) AS trip_count,
       ROUND(AVG(duration_min), 2) AS mean_min,
       ROUND(AVG(duration_min * duration_min) - AVG(duration_min) * AVG(duration_min), 2) AS variance_min2
FROM trips
GROUP BY pickup_zone, dropoff_zone
HAVING COUNT(*) >= 20
ORDER BY variance_min2 DESC
LIMIT 20;

-- Median is not built into standard SQLite; export per-route distributions to Python
-- when comparing tail quantiles, rather than treating mean + sd as a 90th percentile.
SELECT hour, weekday, COUNT(*) AS n,
       ROUND(AVG(duration_min), 2) AS mean_min
FROM trips
GROUP BY hour, weekday
ORDER BY weekday, hour;
