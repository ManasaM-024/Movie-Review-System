# Apache Spark Movie Review System - Report

This repository contains a LaTeX report about implementing Apache Spark in a Movie Review System. The report covers Spark's architecture, components, and how they benefit large-scale movie data processing.

## Report Contents

The report includes:

1. **Introduction** to Apache Spark and its relevance to Movie Review Systems
2. **Spark Architecture and Components**:
   - Spark Core
   - Spark SQL
   - Spark Streaming
   - MLlib
   - GraphX
3. **Implementation Details**:
   - System overview
   - Data processing pipeline
   - Code examples for data transformation and sentiment analysis
   - Real-time processing with Spark Streaming
4. **Performance Benefits** of using Spark
5. **Snapshots and Visualizations** (diagrams and performance comparisons)
6. **Conclusion** and future enhancements

## Using the Report

### Option 1: Overleaf (Recommended)
1. Go to [Overleaf](https://www.overleaf.com)
2. Sign in or create a free account
3. Create a new project
4. Upload `spark_report.tex` to your Overleaf project
5. Compile the document to generate the PDF

### Option 2: Local LaTeX Installation
If you have a LaTeX distribution installed locally:
```bash
pdflatex spark_report.tex
bibtex spark_report.aux
pdflatex spark_report.tex
pdflatex spark_report.tex
```

## Key Features Covered

- How Apache Spark's in-memory computing improves processing speed
- Implementation of sentiment analysis using MLlib
- Real-time review processing with Spark Streaming
- Performance comparisons between Spark and traditional frameworks
- System architecture diagrams

movie-review-system/
├─ README.md                          # Project overview (you already have)
├─ LICENSE
├─ .gitignore
├─ Makefile                            # common tasks (build, test, lint)
├─ requirements.txt                    # python deps (if using PySpark)
├─ docker-compose.yml                  # local dev for kafka/cassandra/etc.
├─ conf/
│  ├─ application.conf.example         # sample app config
│  └─ env.example                       # env var templates
├─ src/
│  ├─ common/
│  │  ├─ __init__.py
│  │  ├─ config.py                      # centralized config loader
│  │  └─ utils.py                       # shared helper functions
│  ├─ streaming/
│  │  ├─ stream_reviews.py              # Spark Structured Streaming job (PySpark)
│  │  ├─ stream_reviews.scala           # (optional) Scala implementation
│  │  ├─ kafka_producers.py             # test producers / sample producers
│  │  └─ checkpoints/                   # (ignored) checkpoint templates
│  ├─ batch/
│  │  ├─ etl_reviews.py                 # batch ETL & parquet conversion
│  │  ├─ train_sentiment_model.py       # batch training job
│  │  └─ retrain_als.py                 # batch job for ALS model
│  ├─ ml/
│  │  ├─ pipeline.py                    # ML pipeline assembly (tokenize→tfidf→model)
│  │  ├─ sentiment/
│  │  │  ├─ train.py
│  │  │  └─ evaluate.py
│  │  └─ recommend/
│  │     ├─ als_train.py
│  │     └─ hybrid_scoring.py
│  └─ preprocessing/
│     ├─ text_cleaning.py
│     └─ tokenizer.py
├─ models/
│  ├─ sentiment/                        # model artifacts (saved pipeline)
│  └─ als/                              # ALS model artifacts
├─ deploy/
│  ├─ docker/
│  │  ├─ Dockerfile.spark
│  │  └─ Dockerfile.worker
│  ├─ k8s/
│  │  ├─ spark-driver.yaml
│  │  └─ spark-executor.yaml
│  ├─ helm/
│  │  └─ movie-review-chart/
│  └─ spark-submit-templates/           # bash templates for submission
├─ examples/
│  ├─ sample_review.json                # example review line
│  ├─ kafka-producer.py                 # simpler sample producer
│  └─ sample_dataset/                   # small CSV/JSON dataset for testing
├─ notebooks/
│  ├─ 01_explore_reviews.ipynb
│  └─ 02_training_experiments.ipynb
├─ scripts/
│  ├─ run_local_stream.sh
│  ├─ run_local_batch.sh
│  └─ deploy_to_k8s.sh
├─ docs/
│  ├─ architecture.md
│  ├─ ops/                              # runbooks, troubleshooting
│  └─ monitoring.md
├─ tests/
│  ├─ unit/
│  │  ├─ test_text_cleaning.py
│  │  └─ test_pipeline.py
│  ├─ integration/
│  │  └─ test_streaming_end2end.py
│  └─ conftest.py
├─ ci/
│  └─ github/
│     └─ workflows/
│        ├─ ci.yml
│        └─ release.yml
└─ examples infra/
   └─ terraform/                        # optional infra-as-code examples

## Prerequisites

For Overleaf:
- Web browser
- Overleaf account

For local compilation:
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages are included in the `.tex` file

## Customization

You can customize the report by:
1. Updating author information
2. Adding your university/institution details
3. Including actual performance metrics from your implementation
4. Adding your own diagrams or screenshots
5. Expanding on specific Spark components relevant to your project

## Related Files

- `spark_report.tex` - Main LaTeX source file
- Diagram files (not included - you can add your own):
  - `spark_architecture.png`
  - `processing_workflow.png`

## References

The report includes academic references to key Spark papers and documentation. All citations follow standard academic formatting.

---
*Note: This report complements the client-side Movie Review System implementation also included in this repository.*