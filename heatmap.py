from main import advertiser_preferences

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

advertiser_preferences = advertiser_preferences.apply(pd.to_numeric, errors='coerce')

# Competition Heatmap: Count the number of non-NaN entries for each billboard
competition = advertiser_preferences.count().sort_index()
plt.figure(figsize=(10, 8))
sns.heatmap(competition.to_frame().T, annot=True, fmt='d', cmap='Reds', cbar_kws={'label': 'Number of Bids'})
plt.title('Billboard Competition Heatmap')
plt.xlabel('Billboard')
plt.ylabel('Number of Bids')
plt.savefig('billboard_competition_heatmap.png', bbox_inches='tight')
plt.show()

# Desirability Heatmap: Calculate the average rank (lower is better) for each billboard
desirability = advertiser_preferences.mean().sort_index()
plt.figure(figsize=(10, 8))
sns.heatmap(desirability.to_frame().T, annot=True, cmap='Blues_r', cbar_kws={'label': 'Average Rank (Lower is Better)'})
plt.title('Billboard Desirability Heatmap')
plt.xlabel('Billboard')
plt.ylabel('Average Rank')
plt.savefig('billboard_desirability_heatmap.png', bbox_inches='tight')
plt.show()

