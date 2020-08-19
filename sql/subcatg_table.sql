CREATE TABLE USER_WORKING.daeyoung_promo_subcatg_wk2 AS (
WITH a AS (
SELECT DISTINCT t.subcatg_id as subcatg_id
FROM line_item_juice s
JOIN taxonomy t
ON s.subcatg_id = t.subcatg_id
WHERE s.sls_trans_dt BETWEEN DATE - INTERVAL '5' YEAR AND DATE - 1
AND s.subcatg_id > 0
),
b AS (
SELECT cal_dt
, cal_yr_num
, qtr_of_yr_num
, mnth_of_yr_num
, promo_wk_num
, CASE WHEN holiday_ind = 'Y' THEN 1 ELSE 0 END AS holiday_ind
, day_of_cal_num
, wk_of_cal_num
, mnth_of_cal_num
, qtr_of_cal_num
, day_of_wk_num
, day_of_mnth_num
, day_of_qtr_num
, day_of_yr_num
, wk_of_mnth_num
, wk_of_yr_num
, mnth_of_qtr_num
FROM ostk_cal
WHERE cal_dt BETWEEN DATE - INTERVAL '5' YEAR AND DATE - 1)
SELECT t.subcatg_id
, t.cal_yr_num
, t.qtr_of_yr_num
, t.mnth_of_yr_num
, t.mnth_of_qtr_num
, t.promo_wk_num
, ZEROIFNULL(spd.site_promo_dsc_amt) AS site_promo_dsc_amt
, MAX(t.holiday_ind) AS holiday_ind
, ZEROIFNULL(SUM(s.line_item_sale)) + ZEROIFNULL(SUM(s.line_item_shpg_amt)) AS gms
, MIN(trans_dt.subcatg_name) as subcatg_name
, MIN(trans_dt.start_dt) as start_dt
, MIN(trans_dt.end_dt) as end_dt
FROM
(SELECT DISTINCT a.subcatg_id
, cal_dt
, promo_wk_num
, wk_of_mnth_num
, cal_yr_num
, qtr_of_yr_num
, mnth_of_yr_num
, mnth_of_qtr_num
, holiday_ind
FROM a,b) t
LEFT JOIN (sel taxonomy.subcatg_id as subcatg_id
			  , taxonomy.subcategory as subcatg_name
			  , min(line_item_juice.sls_trans_dt) as start_dt
			  , max(line_item_juice.sls_trans_dt) as end_dt
			from line_item_juice
			join taxonomy
			on line_item_juice.subcatg_id = taxonomy.subcatg_id
			where taxonomy.subcatg_id > 0
			and line_item_juice.sls_trans_dt BETWEEN DATE - INTERVAL '5' YEAR AND DATE - 1
			group by 1,2) trans_dt
ON t.subcatg_id = trans_dt.subcatg_id
LEFT JOIN line_item_juice s
ON  t.subcatg_id = s.subcatg_id
AND t.cal_dt = s.sls_trans_dt
LEFT JOIN  user_working.line_site_promo_id u
ON u.sls_trans_line_id = s.sls_trans_line_id
AND u.sls_trans_id = s.sls_trans_id
LEFT JOIN site_promo_dsc spd
ON  u.src_site_promo_id = spd.site_promo_id
GROUP BY 1,2,3,4,5,6,7--,8,9,10,11
) WITH DATA PRIMARY INDEX (subcatg_id);




--
--
-- --  when changing month, week, daily, change line 22-24 and last line mnth, wk, cal_dt.
-- CREATE TABLE USER_WORKING.daeyoung_promo_subcatg_daily AS (
-- WITH a AS (
--   SELECT DISTINCT subcatg_id
--   FROM  bi_sls_line_jce_spnd
--   WHERE sls_trans_dt BETWEEN DATE - INTERVAL '2' YEAR AND DATE - 1
--   AND  subcatg_id > 0
--   AND  sls_rpt_std_filter_ind = 'Y'
--   AND  sls_trans_conv_ind = 'Y' ),
--
--   b AS (
--   SELECT cal_dt
--       , wk_of_cal_num
--       , mnth_of_cal_num
--   FROM  ostk_cal
--   WHERE cal_dt BETWEEN DATE - INTERVAL '2' YEAR AND DATE - 1)

-- SELECT t.subcatg_id
--     , t.cal_dt
--     --, wk_of_cal_num
--     --, mnth_of_cal_num
--     , ZEROIFNULL(spd.site_promo_dsc_amt) AS site_promo_dsc_amt
--     , ZEROIFNULL(COUNT(DISTINCT short_sku_item_id)) AS sku_cnt
--     , ZEROIFNULL(SUM(s.item_qty)) AS unit_sold
--     , ZEROIFNULL(SUM(s.gms)) AS gms
--     , ZEROIFNULL(SUM(s.gmv_amt)) AS gmv
--     , ZEROIFNULL(SUM(s.product_juice)) AS juice
--     , ZEROIFNULL(SUM(s.tot_juice)) AS tot_juice
--     , ZEROIFNULL(SUM(s.product_nectar)) AS nectar
--     , ZEROIFNULL(SUM(s.tot_nectar)) AS tot_nectar
--     , ZEROIFNULL(SUM(s.trans_line_dsc_amt)) AS dsc_amt
-- FROM
--     (SELECT a.subcatg_id, b.cal_dt, b.wk_of_cal_num, b.mnth_of_cal_num
--     FROM a,b) t
-- LEFT JOIN bi_sls_line_jce_spnd s
-- ON   t.subcatg_id = s.subcatg_id -- need to ask about how to gather catg level <-- no catg_id in bi_sls_line_jce_spnd table.
-- AND  t.cal_dt = s.sls_trans_dt
-- AND s.sls_rpt_std_filter_ind = 'Y'
-- AND  s.sls_trans_conv_ind = 'Y'
-- LEFT JOIN site_promo_dsc spd
-- ON   s.site_promo_id = spd.site_promo_id
-- GROUP BY 1,2,3
-- ) WITH DATA UNIQUE PRIMARY INDEX (subcatg_id, cal_dt, site_promo_dsc_amt);
--) WITH DATA UNIQUE PRIMARY INDEX (subcatg_id, wk_of_cal_num, site_promo_dsc_amt);
--) WITH DATA UNIQUE PRIMARY INDEX (subcatg_id, mnth_of_cal_num, site_promo_dsc_amt);
