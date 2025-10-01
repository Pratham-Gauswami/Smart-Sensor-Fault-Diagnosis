# 🔧 Smart Sensor Fault Diagnosis & Knowledge Graph System

> An intelligent preventive maintenance pipeline that transforms sensor data chaos into actionable insights using AI-powered fault detection and knowledge graphs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/Neo4j-4.0+-green.svg)](https://neo4j.com/)

---

## 🎯 What is This?

Imagine having a digital expert that never sleeps, constantly monitoring your industrial sensors, predicting failures before they happen, and suggesting exactly how to fix them. That's what this system does.

By combining natural language processing, graph databases, and retrieval-augmented generation, we've built a maintenance copilot that learns from your sensor data and maintenance history to keep your operations running smoothly.

---

## ✨ Key Features

### 🤖 Intelligent Processing
- **Dual Data Ingestion**: Handles both structured sensor readings (CSV, JSON) and unstructured maintenance logs
- **Smart Entity Extraction**: Automatically identifies sensor IDs, fault types, timestamps, and relationships from raw text

### 🕸️ Knowledge Graph Power
- **Relationship Mapping**: Visualize connections between sensors, faults, and maintenance actions
- **Pattern Discovery**: Uncover which sensors frequently fail together or what conditions lead to breakdowns
- **Query Intelligence**: Ask complex questions like "Show me all temperature sensors that failed after pressure spikes"

### 💡 AI-Driven Recommendations
- **RAG-Powered Insights**: Context-aware fault diagnosis using Retrieval-Augmented Generation
- **Actionable Repairs**: Get specific, prioritized repair recommendations based on historical data
- **Preventive Alerts**: Catch issues before they become costly failures

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Data Sources   │
│  • Sensors      │
│  • Log Files    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ NLP Processing  │
│  • Entity Ext.  │
│  • Text Parse   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Knowledge Graph │
│    (Neo4j)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Engine     │
│  • Retrieval    │
│  • Generation   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Recommendations │
│  & Reports      │
└─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher
- Neo4j Desktop or Community Edition
- 4GB RAM minimum (8GB recommended)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-repo/smart-sensor-fault-diagnosis.git
cd smart-sensor-fault-diagnosis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure Neo4j**
- Launch Neo4j and create a new database
- Update credentials in `config/neo4j_config.yaml`:
```yaml
uri: bolt://localhost:7687
username: neo4j
password: your_password
```

**4. Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

**5. Run the system**
```bash
python main.py
```

---

## 📖 Usage Guide

### Preparing Your Data

**Structured Data** (place in `data/structured/`)
```csv
sensor_id,timestamp,temperature,pressure,vibration,status
S1023,2024-01-15 08:30:00,85.2,120.5,0.02,normal
S1023,2024-01-15 09:00:00,92.8,125.3,0.15,warning
```

**Unstructured Logs** (place in `data/unstructured/`)
```text
[2024-01-15 09:15:00] Sensor S1023 reported overheating condition
[2024-01-15 09:20:00] Technician replaced thermal paste on S1023
[2024-01-15 09:45:00] S1023 returned to normal operation
```

### Running the Pipeline

**Basic usage:**
```python
from pipeline import SensorFaultPipeline

# Initialize the pipeline
pipeline = SensorFaultPipeline(
    structured_data_path="data/structured/sensor_readings.csv",
    unstructured_data_path="data/unstructured/logs.txt",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your_password"
)

# Execute the analysis
pipeline.run()
```

**Output example:**
```
╔═══════════════════════════════════════════════════════╗
║           FAULT DIAGNOSIS REPORT                      ║
╚═══════════════════════════════════════════════════════╝

📍 Sensor ID: S1023
🔴 Fault Type: Overheating
📊 Confidence: 94%
⏰ Detected: 2024-01-15 09:15:00

🔧 RECOMMENDED ACTIONS:
1. Inspect cooling system for blockages
2. Replace thermal paste on heat sink
3. Check ambient temperature conditions
4. Verify fan operation

📈 HISTORICAL CONTEXT:
- Similar fault occurred 3 times in past 6 months
- Average repair time: 45 minutes
- Related sensors: S1024, S1025 (same cooling zone)
```

### Querying the Knowledge Graph

Use Neo4j Browser or provided scripts:

```python
# Find sensors with frequent failures
python scripts/neo4j_queries.py --query frequent_failures

# Explore sensor relationships
python scripts/neo4j_queries.py --query related_sensors --sensor_id S1023
```

**Example Cypher query:**
```cypher
MATCH (s:Sensor)-[:HAS_FAULT]->(f:Fault)
WHERE f.type = 'Overheating'
RETURN s.id, count(f) as fault_count
ORDER BY fault_count DESC
LIMIT 10
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.8+ | Core development |
| **NLP** | spaCy, Transformers | Entity extraction & text processing |
| **Database** | Neo4j | Knowledge graph storage |
| **RAG Framework** | LangChain | Context-aware generation |
| **Data Processing** | Pandas, NumPy | Data manipulation & analysis |
| **Deployment** | Docker | Containerized deployment |

---

<!-- ## 📁 Project Structure

```
smart-sensor-fault-diagnosis/
├── config/
│   ├── neo4j_config.yaml      # Database configuration
│   └── model_config.yaml      # NLP model settings
├── data/
│   ├── structured/            # Sensor readings (CSV/JSON)
│   └── unstructured/          # Log files (TXT)
├── outputs/
│   ├── recommendations/       # Generated reports
│   └── graphs/                # Visualization exports
├── scripts/
│   ├── neo4j_queries.py      # Pre-built graph queries
│   └── data_validation.py    # Input data validators
├── src/
│   ├── nlp/                  # NLP processing modules
│   ├── graph/                # Neo4j interactions
│   └── rag/                  # RAG pipeline
├── tests/                    # Unit and integration tests
├── .env.example             # Environment template
├── requirements.txt         # Python dependencies
├── main.py                  # Entry point
└── README.md               # You are here!
```

--- -->

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit** your changes
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push** to your branch
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open** a Pull Request

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact & Support

- **Issues**: Open an issue on [GitHub Issues](https://github.com/your-repo/smart-sensor-fault-diagnosis/issues)
- **Email**: prathamg2612@gmail.com
<!-- - **Documentation**: [Full docs](https://github.com/your-repo/smart-sensor-fault-diagnosis/wiki) -->

---

## 🌟 Star Us!

If this project helps you maintain your industrial systems, please consider giving it a star ⭐ on GitHub. It helps others discover the project!

---

<div align="center">
Made with ❤️ for smarter industrial maintenance
</div>