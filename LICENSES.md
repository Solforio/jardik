# Data source licences

Every source used by Jardik is listed here with its terms and commercial status. Update this file in
the same change that adds a source or dataset. Terms below are marked `TO VERIFY` until checked against
the provider's current official page; nothing here is legal advice.

Before any commercial use: review the licence of the data and of models trained on it (non-commercial
data can restrict the use of trained weights).

| Source | Type | Used for | Terms / URL | Commercial use | Verified on | Status |
|---|---|---|---|---|---|---|
| Building Data Genome 2 (BDG2) | Dataset | Forecasting training (M2) | Repository LICENSE file: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0). Citation: Miller et al., Scientific Data 7, 368 (2020), doi 10.1038/s41597-020-00712-x. Files are stored with Git LFS and checked by SHA-256 | Allowed by the licence WITH attribution and ShareAlike: adaptations (including our extractions) must be shared under the same licence. Whether ShareAlike reaches models trained on it is UNCLEAR: review before any commercial use. One secondary source (BuildingsBench) calls it MIT; we follow the stricter repository LICENSE | 2026-10-16 | In use (prototype) |
| Open-Meteo | Weather API (archive = reanalysis; forecast) | Weather history and forecast (M1) | https://open-meteo.com/en/terms : free API for non-commercial use only; data CC BY 4.0 (attribution required); free limits under 10,000 calls/day, 5,000/hour, 600/minute | **No** on the free tier. A paid plan is required for any commercial use | 2026-10-15 | In use (prototype) |
| AEMET OpenData | Weather API | Spanish weather | TO VERIFY (free API key) | TO VERIFY | - | Planned |
| ERA5 (Copernicus CDS) | Reanalysis | Weather history | TO VERIFY (free account) | TO VERIFY | - | Planned |
| NASA POWER | Weather / solar | Weather | TO VERIFY | TO VERIFY | - | Planned |
| PVGIS (EU JRC) | Solar | PV and irradiance | TO VERIFY | TO VERIFY | - | Planned |
| ESIOS (Red Electrica) | Market | Spanish prices | TO VERIFY (free token) | TO VERIFY | - | Planned |
| ENTSO-E Transparency | Market | European prices | TO VERIFY (free token) | TO VERIFY | - | Planned |
| Datadis | Client data API | Customer smart-meter data | TO VERIFY (needs customer authorisation) | TO VERIFY | - | Planned |
| GoiEner | Dataset | Spanish smart meters (M2b) | TO VERIFY access and licence | TO VERIFY | - | Candidate |
| UK-DALE, REFIT | Dataset | NILM (M5) | TO VERIFY | TO VERIFY | - | Candidate |
