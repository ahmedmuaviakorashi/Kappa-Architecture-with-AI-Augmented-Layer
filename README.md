# Project Report: A Modern Data Architecture for AI-Powered Analytics

**Subject:** Technical Feasibility of a Kappa Architecture with AI-Augmented Feature Store for Large-Scale Data Segmentation  
**GitHub:** [Kappa-Architecture-with-AI-Augmented-Layer](https://github.com)  

---

## 1.0 Executive Summary
This proof-of-concept (POC) evaluates a hybrid Kappa Architecture augmented with an AI layer and Feature Store for large-scale data segmentation. Using NYC taxi trip data (~2M sampled records), the architecture demonstrates that precomputing clusters and AI-enhanced metrics via a stream-processing paradigm is feasible, memory-efficient, and performant.

**Key outcomes:**
- Stream processing of 1.95M records completed successfully with embeddings stored in ChromaDB and features upserted to SQLite online store.
- Precomputed clustering reduced processing time by ~12% compared to on-the-fly computation (1,241.88s vs 1,413.97s) while maintaining low memory footprint (~85–88 MB peak).
- Vector similarity search enabled AI-native queries for semantically similar trips.
- AI-enhanced metrics such as anomaly scores, revenue-per-minute, and silhouette scores were computed efficiently on sampled embeddings.
- The architecture provides a robust foundation for future AI/ML workloads while enabling high-performance analytics and real-time serving.

---

## 2.0 Core Architectural Analysis: Kappa vs. Alternatives
Large-scale clustering and segmentation directly on SQL databases is computationally prohibitive. Iterative algorithms like K-Means are O(n³), making in-database computation for millions of rows inefficient and inflexible.

### 2.1 Problems with SQL-Only Computation
- **Computational Load:** Complex ML algorithms consume database CPU, causing latency spikes.
- **Data Movement:** Full-table computations require large memory/disk allocations.
- **Inflexibility:** Updating clustering models requires full recomputation, leading to downtime or stale data.

### 2.2 Evaluated Architectural Patterns

| Architecture | Description | Pros | Cons |
|--------------|------------|------|------|
| Lambda | Separate Batch (accuracy) and Speed (latency) layers | Handles both historical and real-time well | High complexity; maintaining dual codebases; merging results non-trivial |
| Kappa (Proposed) | Single stream-processing layer; historical data reprocessed from event log | One codebase; inherent reprocessing; flexible AI integration | Requires durable event log; initial full-history processing can be slow |
| Micro-Batch (Spark) | Processes small discrete batches (e.g., 5 min) | Balance between latency and throughput | Higher latency than streaming; management complexity for AI pipelines |

**Rationale:** The Kappa Architecture reduces maintenance complexity and provides flexibility for AI/ML model updates. Replayable event logs and decoupled processing ensure continuous operation without downtime.

---

## 3.0 POC Workflow Deep Dive: Procedures and Justifications

### 3.1 Data Ingestion & Event Log Creation
- NYC taxi data partitioned into 100,000-row chunks and stored as Parquet files (`event_log` directory).  
- Immutable, columnar Parquet format ensures efficient reading and preserves raw data for reprocessing.  

**Why:** Enables true reprocessing of historical events without risk of data loss.

### 3.2 Feature Engineering & Model Training
- Features: `trip_distance`, `total_amount`, `pickup_hour`, `pickup_day`, `trip_duration`, `passenger_count`, discretized pickup/dropoff locations.
- MiniBatchKMeans trained on historical data in batches of 50,000 records to reduce memory footprint.  

**Why MiniBatchKMeans:** Iterative partial fits approximate full K-Means clustering efficiently for millions of rows while controlling memory usage.  
**AI Integration:** `cluster_id` serves as the primary precomputed metric for downstream analytics.

### 3.3 Feature Store: Online & Offline
- **Offline Store:** Parquet files (`offline_features`) containing full historical trips and cluster assignments for model training and batch analytics.  
- **Online Store:** SQLite database (`online_features`) storing latest cluster IDs and key metrics per `trip_id` for low-latency retrieval.  

**Why:** Separates heavy AI computation from serving, avoiding database bottlenecks and enabling rapid access to precomputed features.

### 3.4 Vector Database for Advanced AI Queries
- PCA reduces feature dimensionality (from 8 → 8–128 depending on config).  
- Embeddings stored in ChromaDB with metadata (`cluster_id`, `trip_distance`, `amount`, `pickup_hour`).  
- `vector_similarity_search()` allows retrieval of trips similar to a given trip based on learned feature space.  

**Why:** Enables AI-native semantic search beyond simple SQL queries.

### 3.5 Stream Processing & Reprocessing
- `process_stream_batch()` reads event log chunks, predicts cluster IDs, upserts to online store, stores embeddings in ChromaDB.  
- Demonstrates reprocessing by adjusting cluster count (k) and re-running the pipeline without downtime.  
- Preprocessing and embedding storage handled in batches (5,000–10,000 rows) to maintain low memory footprint.  

**Results:**
- Total processed: 1,956,509 rows  
- Vector DB embeddings stored: 1,956,509  
- Peak memory usage: 88.32 MB  
- Precomputed vs On-the-fly: Precomputed faster by ~172s  

### 3.6 AI-Enhanced Metrics
Computed metrics per trip:  
- **Cluster Size:** Number of trips per cluster  
- **Anomaly Score:** IsolationForest-based detection  
- **Silhouette Score:** Cluster cohesion (~0.054 precomputed, 0.048 on-the-fly)  
- **Revenue per Minute:** Business KPI  
- **Peak Hour Flag:** Indicates rush hour trips  

Metrics saved in `ai_metrics.parquet` and used for AI-driven analytics.

---

## 4.0 Performance Evaluation

| Method | Rows Processed | Time (s) | Peak Memory (MB) |
|--------|----------------|----------|----------------|
| Precomputed | 1,956,509 | 1,241.88 | 85.76 |
| On-the-fly | 1,956,509 | 1,413.97 | 88.32 |

- Precomputed processing is ~12% faster.  
- Silhouette score difference: 0.0067 → negligible, validating precomputed metrics against on-the-fly computation.

---

## 5.0 Key Takeaways & Enhancements
- **Kappa Architecture Feasible:** Handles large-scale segmentation and reprocessing with minimal memory overhead.  
- **AI-Augmented Feature Store:** Separation of online/offline stores enables low-latency lookups while supporting complex analytics.  
- **Vector-Based Search:** PCA embeddings in ChromaDB provide powerful AI-native retrieval capabilities.  
- **Stream-First Processing:** Event log-driven design allows seamless updates and reprocessing.  
- **Memory-Efficient Scaling:** MiniBatchKMeans and chunked stream processing enable operations on limited hardware (16GB RAM).

**Future Enhancements:**
- Integrate a true streaming layer (Kafka, Pulsar) for real-time ingestion.  
- Automate cluster re-evaluation based on new data patterns.  
- Add more business-centric embeddings for recommendation systems.  
- Expand anomaly detection to real-time monitoring.

---

## 6.0 Citations & References
- Kreps, J. (2014). *Questioning the Lambda Architecture*. O'Reilly Radar.  
- Scikit-learn Developers. (2023). *Clustering — scikit-learn 1.3.0 documentation*. scikit-learn.org.  
- He, S. et al. (2021). *What is a Feature Store?* Tecton Blog.

---

## Appendix: Key Metrics Observed
- Total processed trips: 1,956,509  
- Stored embeddings: 1,956,509  
- Peak memory: 88.32 MB  
- Silhouette score: 0.054 (precomputed) / 0.048 (on-the-fly)  
- Vector similarity query: returned 5 most similar trips with meaningful metadata
