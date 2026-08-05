# ATLAS

This section documents the ATLAS-specific data access and preprocessing workflow used in this repository.

The broader goal is to support time-series anomaly detection on ATLAS operational data from monitoring compute nodes for data quality management. The intent is to detect abnormal behavior in the operational stack, including issues affecting compute nodes, data flow, and networking, so that problems can be identified and corrected more quickly. Faster detection helps reduce operational impact and improve the overall efficiency of the experiment.

The current documentation focuses on the data ingestion and alignment layer that prepares run-based PBeast data for downstream analysis.

## Contents

- [`pbeast_fetcher.md`](./pbeast_fetcher.md): overview of the local PBeast fetch package, its configuration model, and how run-based data retrieval works in this repository
- [`time_series_alignment.md`](./time_series_alignment.md): overview of the reference-timeline alignment step used to convert fetched ATLAS time series into a single merged dataset for analysis
