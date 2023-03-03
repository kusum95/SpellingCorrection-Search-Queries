#the query to create brand_freq.csv
#output is a table with brand name and its total sales in the past 2 years

select
    brand_n as brand,
    sum(sale_q) as quantity
from
(select 
  a.tcin,
  a.sale_q,
  b.brand_n
from prd_sls_fnd.sales_line_item a
inner join prd_itm_fnd.item b
on a.tcin = b.tcin
where a.sale_d > date('2021-01-01')
      and (a.customer_order_type='SALES' OR a.customer_order_type='sales')
      and a.transaction_type_c IN ('00', '01')
      and b.brand_n is not null) t
group by brand_n
order by sum(sale_q) desc
