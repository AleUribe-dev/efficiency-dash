# Engineer Efficiency Dashboard

Interactive dashboard built with **Streamlit** to analyze engineers' task efficiency using data exported from task management systems.

---

## ✨ Features

- **Task URL integration**  
  - All tasks get a unique clickable link, built from `ID` + a sidebar URL template.  
  - Links displayed as a small 🔗 icon next to the task name (clean, no raw URLs).

- **Interactive tables**  
  - Powered by `st.dataframe`.  
  - Support for quick filtering, sorting, copying.  
  - Consistent UI across all panels.

- **General Performance**
  - **Task Difficulty Distribution** with tabs:  
    - **Chart**: distribution plot of task difficulties.  
    - **Table**: sortable task list with 🔗 links.  
  - **Tasks with DOC blocks (details)**: quick filtered table with task links.  
  - Activity type distributions, heatmaps, top-5 charts.

- **Individual Performance**
  - Tabs per engineer:
    - **Task Difficulty Distribution** (Chart / Table).  
    - Scatterplots of durations, activity insights, and linked tasks.  
  - ZIP export per engineer (with traceability of thresholds).

- **Robust metrics & scoring**
  - Composite Difficulty Score from:
    - Task depth.  
    - DOC activity weight.  
    - Task duration.  
  - Robust stats (median/IQR) instead of simple mean.  
  - Post-classification micro-gating with global strictness and minimum signals.

- **Configurable parameters**
  - Sidebar controls for DOC activity weights, thresholds, and Task link template.  
  - All defaults preloaded, but adjustable on the fly.

---

## 🛠 Tech Stack

- **Language**: Python 3.x  
- **Frontend**: Streamlit  
- **Visualization**: Plotly Express  
- **Data**: Pandas, NumPy  
- **Export**: ZIP reports per engineer

---

## 🚀 Usage

1. Clone this repo.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Upload your **Task Management export (Excel)**.
5. Adjust sidebar parameters (DOC weights, thresholds, Task link template).
6. Explore the **General** and **Individual** performance tabs.

---

## 📊 Example Workflow

- Upload export file `2025 Task Management Export.xlsx`.
- In the sidebar:
  - Set `ID column = ID`.  
  - Set `Task link template = https://miempresa.com/tasks/{id}`.  
- Navigate to:
  - **General Performance** → view difficulty distribution and task tables.  
  - **Individual Performance** → deep-dive into each engineer's performance.  
- Click the 🔗 icon to jump directly to a task in your tracker.

---

## 📌 Backlog / Nice to Have

- Clickable points in graphs (e.g. “Task duration over time”) to jump directly to tasks.  
- Option to embed the link directly in the task name (requires HTML rendering, loses sorting/filter).  
- Further UI polish (headers, icons, compact layouts).

---

## 👨‍💻 Author

Developed by Alejandro Uribe & GPT-5 (OpenAI) as part of an **Engineer Efficiency Model** initiative.  
Focus: empowering teams with data-driven insights into task management and productivity.
