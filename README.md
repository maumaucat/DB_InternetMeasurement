# Evaluating Mobile Connectivity Across Providers in Train Environments

## Scripts

- **`netzlog.sh`**  
  Bash script to measure round-trip times (RTT) using `ping`, triggered based on GPS accuracy and train movement speed.

- **`plot.py`**  
  Python script for visualizing different types of data collected by the Breitbandmessung app and the Bash script.

- **`aggregate.py`**  
  Utility functions for flexible aggregation of collected data points.

---

## Data Collection

Measurements were conducted along the following train routes:

- **Osnabrück – Münster**  
- **Münster – Enschede**  
- **Hengelo – Bielefeld**

### Data Sources

- **[Breitbandmessung](https://breitbandmessung.de/)** app  
  Official German broadband measurement application for mobile and stationary networks.

- **`netzlog.sh`**  
  Custom logging script used for measuring network latency in real-time train environments.

---

## Further Information

For more details on methodology, evaluation, and results, please refer to the [report](./).
