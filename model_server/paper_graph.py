import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the table
# Note: AUC_ROC is excluded as it's not in the reference bar chart.
models = [
    'Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN',
    'XGBoost', 'Neural Network', 'Naive Bayes'
]

# Metrics data
accuracy = [0.87, 0.96, 0.98, 0.83, 0.979, 0.969, 0.87]
precision = [0.77, 0.95, 0.95, 0.72, 0.943, 0.899, 0.73]
recall = [0.86, 0.93, 0.97, 0.75, 0.945, 0.94, 0.91]
f1_score = [0.81, 0.94, 0.96, 0.74, 0.944, 0.919, 0.81]

# Colors from the reference image
colors = {
    'Accuracy': '#5eb13d',  # Green
    'Precision': '#439bda', # Blue
    'Recall': '#7e28a5',   # Purple
    'F1 Score': '#fec104'  # Yellow/Gold
}


# --- Plotting ---
x = np.arange(len(models))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8))

# Creating the bars for each metric
rects1 = ax.bar(x - 1.5 * width, accuracy, width, label='Accuracy', color=colors['Accuracy'])
rects2 = ax.bar(x - 0.5 * width, precision, width, label='Precision', color=colors['Precision'])
rects3 = ax.bar(x + 0.5 * width, recall, width, label='Recall', color=colors['Recall'])
rects4 = ax.bar(x + 1.5 * width, f1_score, width, label='F1 Score', color=colors['F1 Score'])

# Add some text for labels, title and axes ticks
ax.set_ylabel('Scores', fontsize=15)
ax.set_xlabel('Models', fontsize=15)
ax.set_title('Comparison of Metrics Across Models', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha="right",fontsize=14.1)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=12)


# Set Y-axis limits to match the reference image
ax.set_ylim(0, 1.05)

# Adjust layout to make room for rotated x-tick labels
fig.tight_layout()

# Display the plot
plt.show()