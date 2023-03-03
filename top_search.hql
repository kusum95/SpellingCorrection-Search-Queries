-- Top 100,000 corrected search queries

SELECT 
	search_term_ignore_case,
	count(1) as frequencies 
FROM prd_real_fnd.real_search_metrics_batch_prod
WHERE search_term_ignore_case != "non-search" 
GROUP BY search_term_ignore_case 
ORDER BY frequencies DESC 
LIMIT 100000;
