import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos de los archivos CSV correctos
scores = pd.read_csv('training/v3_dqn_dqps_1500_0.001_epsilon_arreglado/summary_scores.csv')  # archivo con 'score'
averaged_scores = pd.read_csv('training/v3_dqn_dqps_1500_0.001_epsilon_arreglado/summary_averaged_scores.csv')  # archivo con 'average_score'
epsilons = pd.read_csv('training/v3_dqn_dqps_1500_0.001_epsilon_arreglado/summary_epsilons.csv')  # archivo con 'epsilon'

# Gráfica de recompensas acumuladas por episodio
plt.figure(figsize=(12,6))
plt.plot(scores['episode'], scores['score'], label='Recompensa por episodio', alpha=0.3)
plt.plot(averaged_scores['episode'], averaged_scores['average_score'], label='Recompensa media móvil (50 episodios)', linewidth=2)
plt.xlabel('Episodio')
plt.ylabel('Recompensa acumulada')
plt.title('Evolución de la recompensa acumulada durante el entrenamiento')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('reward_evolution.png')  # Guardar figura
plt.show()

# Gráfica de evolución de epsilon
plt.figure(figsize=(12,4))
plt.plot(epsilons['episode'], epsilons['epsilon'], color='orange')
plt.xlabel('Episodio')
plt.ylabel('Epsilon (exploración)')
plt.title('Evolución de la tasa de exploración $\epsilon$ durante el entrenamiento')
plt.grid(True)
plt.tight_layout()
plt.savefig('epsilon_evolution.png')
plt.show()
