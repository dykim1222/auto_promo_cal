-- catg
-- monthly

CREATE TABLE USER_WORKING.daeyoung_promo_catg_wk AS (
WITH a AS (
SELECT DISTINCT t.Catg_ID AS catg_id
, s.subcatg_id
FROM line_item_juice s
JOIN taxonomy t
ON  s.subcatg_id = t.SubCatg_ID
WHERE sls_trans_dt BETWEEN DATE - INTERVAL '5' YEAR AND DATE - 1
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
SELECT t.catg_id
, t.cal_yr_num
, t.qtr_of_yr_num
, t.mnth_of_yr_num
, t.mnth_of_qtr_num
, t.promo_wk_num
, ZEROIFNULL(spd.site_promo_dsc_amt) AS site_promo_dsc_amt
, MAX(t.holiday_ind) AS holiday_ind
, ZEROIFNULL(SUM(s.line_item_sale)) + ZEROIFNULL(SUM(s.line_item_shpg_amt)) AS gms
, MIN(trans_dt.catg_name) as catg_name
, MIN(trans_dt.start_dt) as start_dt
, MIN(trans_dt.end_dt) as end_dt
FROM
(SELECT DISTINCT a.catg_id, a.subcatg_id, b.cal_dt, b.wk_of_cal_num, b.mnth_of_cal_num
, cal_yr_num
, qtr_of_yr_num
, mnth_of_yr_num
, mnth_of_qtr_num
, qtr_of_cal_num
, promo_wk_num
, holiday_ind
FROM a,b) t
LEFT JOIN (sel taxonomy.catg_id as catg_id
			  , taxonomy.category as catg_name
			  , min(line_item_juice.sls_trans_dt) as start_dt
			  , max(line_item_juice.sls_trans_dt) as end_dt
			from line_item_juice
			join taxonomy
			on line_item_juice.subcatg_id = taxonomy.subcatg_id
			where taxonomy.catg_id > 0
			and line_item_juice.sls_trans_dt BETWEEN DATE - INTERVAL '5' YEAR AND DATE - 1
			group by 1,2) trans_dt
ON t.catg_id = trans_dt.catg_id
LEFT JOIN line_item_juice s
ON  t.subcatg_id = s.subcatg_id
AND t.cal_dt = s.sls_trans_dt
LEFT JOIN  user_working.line_site_promo_id u --Komson generated this for me
ON u.sls_trans_line_id = s.sls_trans_line_id
AND u.sls_trans_id = s.sls_trans_id
LEFT JOIN site_promo_dsc spd
ON  u.src_site_promo_id = spd.site_promo_id
GROUP BY 1,2,3,4,5,6,7--,8,9,10,11
) WITH DATA PRIMARY INDEX (catg_id);

update USER_WORKING.daeyoung_promo_catg_wk set catg_name = oreplace(catg_name,',', ' ') where catg_name like '%,%'
