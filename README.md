### Project Title

**Statistical Simulation and Validation of LDA Topic Models using MySQL**

### Project Overview

This **Final Project** for the **Statistical Simulation and Database Programming** course implements a complete pipeline to **rigorously validate** the performance of the Latent Dirichlet Allocation (LDA) Topic Model under controlled, simulated conditions.

### Core Objectives

1.  **Statistical Simulation:** Implement the **Dirichlet-Multinomial** generative process to create a large-scale (5,000 documents) synthetic text corpus. Crucially, the **Ground Truth** document-topic distribution ($\theta_{true}$) is known.
2.  **Database Engineering:** Design and implement a robust MySQL database schema with three tables (`sim_parameters`, `documents_data`, `analysis_results`) to manage both the non-structured text data and the structured statistical vectors ($\theta_{true}$).
3.  **Model Validation (Text Mining):** Apply the LDA algorithm to the database-stored corpus to infer the predicted distribution ($\theta_{pred}$).
4.  **Statistical Analysis:** Quantify model performance by calculating statistical metrics like **Perplexity** and the **Cosine Similarity** between the known $\theta_{true}$ and the inferred $\theta_{pred}$.

### Technologies Used

| Category | Tools/Languages |
| :--- | :--- |
| **Statistical Programming** | Python (NumPy, SciPy) |
| **Text Mining/ML** | Python (Gensim/scikit-learn) |
| **Database Management** | MySQL, SQL, Python (PyMySQL) |
| **Data Visualization** | Python (Matplotlib/Seaborn) |

