# ClusteringBoutiqueBookstore

# 📊 Book Purchase Analysis & Customer Segmentation

## 📌 Overview  
This project explores **book purchase patterns** using **cross-correlation analysis** and applies **Agglomerative Clustering & KMeans** to segment customers based on shopping behavior. The results provide insights for **feature selection** and **market segmentation** in book sales.

## 🚀 Features  
- **📈 Cross-Correlation Analysis:** Identifies relationships between book genres to optimize feature selection.  
- **📊 Agglomerative Clustering:** Segments customers using hierarchical clustering with **Manhattan distance**.  
- **🔢 KMeans Clustering:** Validates segmentation results by clustering data into **4 distinct groups**.  
- **📉 Dendrogram Visualization:** Illustrates hierarchical merging of the last **20 clusters**.

## 📂 Dataset  
The dataset consists of **1,000 shopping records** across **20 book categories**, tracking purchasing behavior to identify trends.

## 📊 Results  
- **Feature Selection:** Removed **3 weakly correlated attributes** (Art&Hist, Poetry, Gifts).  
- **Strongest Correlation:** **Non-Fiction & Romance (0.82)**, **Manga & Baby Thriller (0.68)**.  
- **Customer Segments Identified:** **4 clusters** with sizes **250, 340, 220, 190**.  

## 🛠 Technologies Used  
- **Python** (Pandas, NumPy, SciPy, Scikit-learn)  
- **Matplotlib & Plotly** (For visualizations)  
