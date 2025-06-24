# **/notebooks/ \- Jupyter Notebooks for Research & Exploration**

This directory serves as the workspace for Jupyter Notebooks used within the Zanzibar Analytics project. Notebooks are an invaluable tool for interactive data analysis, research, algorithm prototyping, visualization, and generating reports.

## **Purpose**

* **Exploratory Data Analysis (EDA):** Investigate raw and processed market data, understand its characteristics, and identify potential patterns or anomalies.  
* **Algorithm Prototyping:** Rapidly develop and test new analytical techniques, indicator calculations, or machine learning models before integrating them into the core Python modules.  
* **Research & Experimentation:** Conduct research on market phenomena, backtest strategy ideas at a conceptual level, and experiment with different parameters or approaches.  
* **Visualization:** Create charts and plots to visualize data, model outputs, and analytical results.  
* **Reporting & Presentation:** Generate reports or presentations summarizing findings from research or analysis.  
* **Educational Material:** Can be used to document specific analytical methods or demonstrate how to use parts of the Zanzibar system.

## **Structure**

It's recommended to organize notebooks into subdirectories based on their purpose:

* **research/**: For ongoing research, development of new ideas, and deep dives into specific analytical problems.  
  * Example: research/wyckoff\_event\_vsa\_deep\_dive.ipynb, research/alternative\_delta\_heuristics.ipynb.  
* **reports/**: For notebooks that produce specific, shareable reports or visualizations.  
  * Example: reports/weekly\_xauusd\_phase\_analysis.ipynb.  
* **prototypes/**: For early-stage, quick prototypes of new features or algorithms.  
* **tutorials/**: (Future) Notebooks demonstrating how to use specific Zanzibar modules or features.

## **Best Practices for Notebooks**

* **Clear Naming:** Use descriptive filenames for notebooks.  
* **Markdown Cells:** Utilize Markdown cells extensively for explanations, comments, and structuring the notebook's narrative.  
* **Code Modularity:** Where possible, encapsulate reusable code into functions within the notebook or, if stable, move it to the core zanzibar/ package and import it.  
* **Environment Consistency:** Ideally, notebooks should be run within the project's Python virtual environment to ensure access to the same dependencies as the main application.  
* **Version Control:** Commit notebooks to Git, but be mindful of large output cells. Consider clearing outputs before committing or using tools like nbstripout to manage this.  
* **Data Paths:** Use relative paths (e.g., ../data/your\_file.csv) to access data files from the /data/ directory, assuming notebooks are run from their location within /notebooks/.  
* **Kernel:** Ensure notebooks use the project's virtual environment kernel.

This directory is a vital space for innovation and exploration within the Zanzibar Analytics project.