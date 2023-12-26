USE [hibernia]
GO

/****** Object:  View [dbo].[real_estate_ro]    Script Date: 23/05/2023 19:21:57 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


ALTER VIEW [dbo].[real_estate_ro] AS
SELECT *
FROM dbo.Real_estate_01_Apr_2023

UNION ALL
SELECT *
FROM dbo.Real_estate_01_May_2023

UNION ALL
SELECT *
FROM dbo.Real_estate_03_Apr_2023

UNION ALL
SELECT id, judet, localitate, zona, commission, latitudine, longitudine, price_EUR, rooms, construction_year, useful_surface, floor_level, 
	   date_of_listing, pret_negociabil, nr_of_kitchens, nr_bai, nr_balcoane, nr_garaje, partitioning, comfort, structura_rezistenta, tip_imobil, 
	   nr_locuri_parcare, furnishing, link_detalii, tip_client, [agent.nume], [agent.functie], [agent.firma], date_snapshot, under_construction_y_n, 
	   project_phase_y_n, max_floor, mansard_y_n, termoficare_y_n, centrala_proprie_y_n, centrala_imobil_y_n, floor_heating_y_n, days_since_listing, 
	   month_of_listing, year_of_listing, price_EUR_sqm
FROM dbo.Real_estate_07_Feb_2023

UNION ALL
SELECT id, judet, localitate, zona, commission, latitudine, longitudine, price_EUR, rooms, construction_year, useful_surface, floor_level, 
	   date_of_listing, pret_negociabil, nr_of_kitchens, nr_bai, nr_balcoane, nr_garaje, partitioning, comfort, structura_rezistenta, tip_imobil, 
	   nr_locuri_parcare, furnishing, link_detalii, tip_client, [agent.nume], [agent.functie], [agent.firma], date_snapshot, under_construction_y_n, 
	   project_phase_y_n, max_floor, mansard_y_n, termoficare_y_n, centrala_proprie_y_n, centrala_imobil_y_n, floor_heating_y_n, days_since_listing, 
	   month_of_listing, year_of_listing, price_EUR_sqm
FROM dbo.Real_estate_07_Mar_2023

UNION ALL
SELECT *
FROM dbo.Real_estate_08_May_2023

UNION ALL
SELECT *
FROM dbo.Real_estate_10_Apr_2023

UNION ALL
SELECT id, judet, localitate, zona, commission, latitudine, longitudine, price_EUR, rooms, construction_year, useful_surface, floor_level, 
	   date_of_listing, pret_negociabil, nr_of_kitchens, nr_bai, nr_balcoane, nr_garaje, partitioning, comfort, structura_rezistenta, tip_imobil, 
	   nr_locuri_parcare, furnishing, link_detalii, tip_client, [agent.nume], [agent.functie], [agent.firma], date_snapshot, under_construction_y_n, 
	   project_phase_y_n, max_floor, mansard_y_n, termoficare_y_n, centrala_proprie_y_n, centrala_imobil_y_n, floor_heating_y_n, days_since_listing, 
	   month_of_listing, year_of_listing, price_EUR_sqm
FROM dbo.Real_estate_13_Feb_2023

UNION ALL
SELECT id, judet, localitate, zona, commission, latitudine, longitudine, price_EUR, rooms, construction_year, useful_surface, floor_level, 
	   date_of_listing, pret_negociabil, nr_of_kitchens, nr_bai, nr_balcoane, nr_garaje, partitioning, comfort, structura_rezistenta, tip_imobil, 
	   nr_locuri_parcare, furnishing, link_detalii, tip_client, [agent.nume], [agent.functie], [agent.firma], date_snapshot, under_construction_y_n, 
	   project_phase_y_n, max_floor, mansard_y_n, termoficare_y_n, centrala_proprie_y_n, centrala_imobil_y_n, floor_heating_y_n, days_since_listing, 
	   month_of_listing, year_of_listing, price_EUR_sqm
FROM dbo.Real_estate_13_Mar_2023

UNION ALL
SELECT *
FROM dbo.Real_estate_17_Apr_2023

UNION ALL
SELECT *
FROM dbo.Real_estate_20_Mar_2023

UNION ALL
SELECT *
FROM dbo.Real_estate_24_Apr_2023

UNION ALL
SELECT *
FROM dbo.Real_estate_27_Mar_2023

UNION ALL
SELECT *
FROM dbo.Real_estate_15_May_2023

UNION ALL
SELECT *
FROM dbo.Real_estate_22_May_2023
GO


