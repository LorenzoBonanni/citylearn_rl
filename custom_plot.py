import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import seaborn as sns
import os
from citylearn.citylearn import CityLearnEnv
from citylearn.building import DynamicsBuilding

def get_kpis(env: CityLearnEnv, standard_baseline_env: CityLearnEnv = None, baseline_condition = None) -> pd.DataFrame:
    """Returns evaluation KPIs.

    Electricity cost and carbon emissions KPIs are provided
    at the building-level and average district-level. Average daily peak,
    ramping and (1 - load factor) KPIs are provided at the district level.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment instance.
    standard_baseline_env: CityLearnEnv, optional
        Standard environment to use as baseline for all environments. 
        If provided, all environments will be compared against this baseline.
    baseline_condition: EvaluationCondition, optional
        Baseline condition to use for evaluation. If None, uses default from CityLearn.

    Returns
    -------
    kpis: pd.DataFrame
        KPI table.
    """
    from citylearn.citylearn import EvaluationCondition

    if standard_baseline_env is None:
        # Use normal evaluation with environment's own baseline
        kpis = env.unwrapped.evaluate(baseline_condition=baseline_condition)
    else:
        if any(isinstance(b, DynamicsBuilding) for b in env.unwrapped.buildings):
            control_condition = EvaluationCondition.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV
            baseline_condition = EvaluationCondition.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV if baseline_condition is None else baseline_condition
        else:
            control_condition = EvaluationCondition.WITH_STORAGE_AND_PV
            baseline_condition = EvaluationCondition.WITHOUT_STORAGE_BUT_WITH_PV if baseline_condition is None else baseline_condition
        
        kpis = env.unwrapped.evaluate(control_condition=control_condition, baseline_condition=baseline_condition)

    kpi_names = {
        'cost_total': 'Cost',
        'carbon_emissions_total': 'Emissions',
        'daily_peak_average': 'Avg. daily peak',
        #'discomfort_proportion': 'Discomfort',
        'discomfort_cold_proportion': 'Discomfort cold',
        'discomfort_hot_proportion': 'Discomfort hot',
        'ramping_average': 'Ramping',
        'monthly_one_minus_load_factor_average': '1 - load factor'
    }
    kpis = kpis[
        (kpis['cost_function'].isin(kpi_names))
    ].dropna()
    kpis['cost_function'] = kpis['cost_function'].map(lambda x: kpi_names[x])

    # round up the values to 2 decimal places for readability
    kpis['value'] = kpis['value'].round(2)

    # rename the column that defines the KPIs
    kpis = kpis.rename(columns={'cost_function': 'kpi'})

    return kpis

def plot_building_kpis(envs: dict[str, CityLearnEnv], standard_baseline_env: CityLearnEnv = None) -> plt.Figure:
    """Plots electricity consumption, cost and carbon emissions
    at the building-level for different control agents in bar charts.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.
    standard_baseline_env: CityLearnEnv, optional
        Standard environment to use as baseline for all environments.
        If provided, all environments will be compared against this baseline.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    kpis_list = []

    for k, v in envs.items():
        kpis = get_kpis(v, standard_baseline_env)
        kpis = kpis[kpis['level']=='building'].copy()
        kpis['building_id'] = kpis['name'].str.split('_', expand=True)[1]
        kpis['building_id'] = kpis['building_id'].astype(int).astype(str)
        kpis['env_id'] = k
        kpis_list.append(kpis)

    kpis = pd.concat(kpis_list, ignore_index=True, sort=False)
    kpi_names = kpis['kpi'].unique()
    column_count_limit = 3
    row_count = math.ceil(len(kpi_names)/column_count_limit)
    column_count = min(column_count_limit, len(kpi_names))
    building_count = len(kpis['name'].unique())
    env_count = len(envs)
    figsize = (3.0*column_count, 0.3*env_count*building_count*row_count)
    fig, axes = plt.subplots(
        row_count, column_count, figsize=figsize, sharey=True
    )
    
    # Appiattire gli assi se necessario per renderli iterabili uniformemente
    if row_count == 1 and column_count == 1:
        axes = np.array([axes])
    elif row_count == 1:
        axes = np.array([axes])
    elif column_count == 1:
        axes = np.array([[ax] for ax in axes])
    
    # Contatore per tenere traccia di quanti KPI sono stati plottati
    plotted_kpis = 0
    
    for i, (k, k_data) in enumerate(kpis.groupby('kpi')):
        # Calcola la posizione dell'asse
        row_idx = i // column_count
        col_idx = i % column_count
        
        if row_count == 1:
            ax = axes[col_idx]
        else:
            ax = axes[row_idx, col_idx]
            
        sns.barplot(x='value', y='name', data=k_data, hue='env_id', ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(k)
        
        plotted_kpis += 1

        for j, _ in enumerate(envs):
            ax.bar_label(ax.containers[j], fmt='%.2f')

        if i == len(kpi_names) - 1:
            legend = ax.legend(
                title="Agenti di controllo",
                loc='upper left', 
                bbox_to_anchor=(1.05, 1.0), 
                framealpha=0.8,
                shadow=True,
                borderpad=1
            )
            plt.setp(legend.get_title(), fontweight='bold')
        else:
            ax.legend().set_visible(False)

        for s in ['right','top']:
            ax.spines[s].set_visible(False)
    
    # Rimuovi gli assi vuoti
    for i in range(plotted_kpis, row_count * column_count):
        row_idx = i // column_count
        col_idx = i % column_count
        
        if row_count == 1:
            ax = axes[col_idx]
        else:
            ax = axes[row_idx, col_idx]
            
        fig.delaxes(ax)
    
    plt.tight_layout()
    return fig

def ensure_dir(directory):
    """Crea la directory se non esiste."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Creata directory: {directory}")

def plot_district_kpis(envs: dict[str, CityLearnEnv], standard_baseline_env: CityLearnEnv = None) -> plt.Figure:
    """Plots electricity consumption, cost, carbon emissions,
    average daily peak, ramping and (1 - load factor) at the
    district-level for different control agents in a bar chart.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.
    standard_baseline_env: CityLearnEnv, optional
        Standard environment to use as baseline for all environments.
        If provided, all environments will be compared against this baseline.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    kpis_list = []

    for k, v in envs.items():
        kpis = get_kpis(v, standard_baseline_env)
        kpis = kpis[kpis['level']=='district'].copy()
        kpis['env_id'] = k
        kpis_list.append(kpis)

    kpis = pd.concat(kpis_list, ignore_index=True, sort=False)
    row_count = 1
    column_count = 1
    env_count = len(envs)
    kpi_count = len(kpis['kpi'].unique())
    figsize = (6.0*column_count, 0.225*env_count*kpi_count*row_count)
    fig, ax = plt.subplots(row_count, column_count, figsize=figsize)
    sns.barplot(x='value', y='kpi', data=kpis, hue='env_id', ax=ax)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    for j, _ in enumerate(envs):
        ax.bar_label(ax.containers[j], fmt='%.2f')

    for s in ['right','top']:
        ax.spines[s].set_visible(False)

    legend = ax.legend(
        title="Agenti di controllo",
        loc='upper left', 
        bbox_to_anchor=(1.05, 1.0), 
        framealpha=0.8,
        shadow=True,
        borderpad=1
    )
    plt.setp(legend.get_title(), fontweight='bold')
    plt.tight_layout()

    return fig

def plot_building_load_profiles(
    envs: dict[str, CityLearnEnv], daily_average: bool = None
) -> plt.Figure:
    """Plots building-level net electricty consumption profile
    for different control agents.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.
    daily_average: bool, default: False
        Whether to plot the daily average load profile.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    daily_average = False if daily_average is None else daily_average
    building_count = len(list(envs.values())[0].buildings)
    column_count_limit = 4
    row_count = math.ceil(building_count/column_count_limit)
    column_count = min(column_count_limit, building_count)
    figsize = (4.0*column_count, 1.75*row_count)
    fig, _ = plt.subplots(row_count, column_count, figsize=figsize)

    for i, ax in enumerate(fig.axes):
        for k, v in envs.items():
            y = v.unwrapped.buildings[i].net_electricity_consumption
            y = np.reshape(y, (-1, 24)).mean(axis=0) if daily_average else y
            x = range(len(y))
            ax.plot(x, y, label=k)

        ax.set_title(v.unwrapped.buildings[i].name)
        ax.set_ylabel('kWh')

        if daily_average:
            ax.set_xlabel('Hour')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

        else:
            ax.set_xlabel('Time step')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(24))

        if i == building_count - 1:
            legend = ax.legend(
                title="Agenti di controllo",
                loc='upper left', 
                bbox_to_anchor=(1.05, 1.0), 
                framealpha=0.8,
                shadow=True,
                borderpad=1
            )
            plt.setp(legend.get_title(), fontweight='bold')
        else:
            ax.legend().set_visible(False)

    plt.tight_layout()

    return fig

def plot_district_load_profiles(
    envs: dict[str, CityLearnEnv], daily_average: bool = None
) -> plt.Figure:
    """Plots district-level net electricty consumption profile
    for different control agents.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.
    daily_average: bool, default: False
        Whether to plot the daily average load profile.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    daily_average = False if daily_average is None else daily_average
    figsize = (5.0, 1.5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for k, v in envs.items():
        y = v.unwrapped.net_electricity_consumption
        y = np.reshape(y, (-1, 24)).mean(axis=0) if daily_average else y
        x = range(len(y))
        ax.plot(x, y, label=k)

    ax.set_ylabel('kWh')

    if daily_average:
        ax.set_xlabel('Hour')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    else:
        ax.set_xlabel('Time step')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))

    legend = ax.legend(
        title="Agenti di controllo",
        loc='upper left', 
        bbox_to_anchor=(1.05, 1.0), 
        framealpha=0.8,
        shadow=True,
        borderpad=1
    )
    plt.setp(legend.get_title(), fontweight='bold')

    plt.tight_layout()
    return fig

def plot_battery_soc_profiles(envs: dict[str, CityLearnEnv]) -> plt.Figure:
    """Plots building-level battery SoC profiles fro different control agents.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    building_count = len(list(envs.values())[0].buildings)
    column_count_limit = 4
    row_count = math.ceil(building_count/column_count_limit)
    column_count = min(column_count_limit, building_count)
    figsize = (4.0*column_count, 1.75*row_count)
    fig, _ = plt.subplots(row_count, column_count, figsize=figsize)

    for i, ax in enumerate(fig.axes):
        for k, v in envs.items():
            y = np.array(v.unwrapped.buildings[i].electrical_storage.soc)
            x = range(len(y))
            ax.plot(x, y, label=k)

        ax.set_title(v.unwrapped.buildings[i].name)
        ax.set_xlabel('Time step')
        ax.set_ylabel('SoC')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
        ax.set_ylim(0.0, 1.0)

        if i == building_count - 1:
            legend = ax.legend(
                title="Agenti di controllo",
                loc='upper left', 
                bbox_to_anchor=(1.05, 1.0), 
                framealpha=0.8,
                shadow=True,
                borderpad=1
            )
            plt.setp(legend.get_title(), fontweight='bold')
        else:
            ax.legend().set_visible(False)

    plt.tight_layout()

    return fig

def plot_simulation_summary(envs: dict[str, CityLearnEnv], standard_baseline_env: CityLearnEnv = None, save_dir='plots'):
    """Plots KPIs, load and battery SoC profiles for different control agents and saves them.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.
    standard_baseline_env: CityLearnEnv, optional
        Standard environment to use as baseline for all environments.
        If provided, all environments will be compared against this baseline.
    save_dir: str, default: 'plots'
        Directory where to save the plot images.
    """
    ensure_dir(save_dir)
    
    print('#'*8 + ' BUILDING-LEVEL ' + '#'*8)
    print('Building-level KPIs:')
    fig = plot_building_kpis(envs, standard_baseline_env)
    plt.savefig(f"{save_dir}/building_kpis.png", bbox_inches='tight', dpi=300)
    plt.show()

    print('Building-level simulation period load profiles:')
    fig = plot_building_load_profiles(envs)
    plt.savefig(f"{save_dir}/building_load_profiles.png", bbox_inches='tight', dpi=300)
    plt.show()

    #print('Building-level daily-average load profiles:')
    #fig = plot_building_load_profiles(envs, daily_average=True)
    #plt.savefig(f"{save_dir}/building_daily_avg_load.png", bbox_inches='tight', dpi=300)
    #plt.show()

    print('Battery SoC profiles:')
    fig = plot_battery_soc_profiles(envs)
    plt.savefig(f"{save_dir}/battery_soc_profiles.png", bbox_inches='tight', dpi=300)
    plt.show()

    print('#'*8 + ' DISTRICT-LEVEL ' + '#'*8)
    print('District-level KPIs:')
    fig = plot_district_kpis(envs, standard_baseline_env)
    plt.savefig(f"{save_dir}/district_kpis.png", bbox_inches='tight', dpi=300)
    plt.show()

    #print('District-level simulation period load profiles:')
    #fig = plot_district_load_profiles(envs)
    #plt.savefig(f"{save_dir}/district_load_profiles.png", bbox_inches='tight', dpi=300)
    #plt.show()

    print('District-level daily-average load profiles:')
    fig = plot_district_load_profiles(envs, daily_average=True)
    plt.savefig(f"{save_dir}/district_daily_avg_load.png", bbox_inches='tight', dpi=300)
    plt.show()

def plot_rewards(ax: plt.Axes, rewards: list[float], title: str) -> plt.Axes:
    """Plots rewards over training episodes.

    Parameters
    ----------
    rewards: list[float]
        List of reward sum per episode.
    title: str
        Plot axes title

    Returns
    -------
    ax: plt.Axes
        Plotted axes
    """

    ax.plot(rewards)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)

    return ax

def plot_reward_summary(envs: dict[str, list[float]], save_dir='plots'):
    """Plotta i reward degli agenti durante il training.
    
    Parameters
    ----------
    envs: dict[str, list[float]]
        Dizionario che mappa il nome dell'agente alla lista dei suoi reward
    save_dir: str, default: 'plots'
        Directory dove salvare l'immagine del grafico
    """
    # Assicurati che la directory esista
    ensure_dir(save_dir)
    
    # Crea la figura con un subplot per ogni agente
    fig, axs = plt.subplots(1, len(envs), figsize=(12, 2))
    
    # Plotta i reward per ogni agente
    for ax, (k, v) in zip(fig.axes, envs.items()):
        ax = plot_rewards(ax, v, k)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/rewards_comparison.png", bbox_inches='tight', dpi=300)
    plt.show()

def plot_post_training_rewards(evaluation_results, save_dir='plots', title_suffix=""):
    """
    Visualizza l'andamento delle ricompense e delle azioni dopo il training.
    
    Parametri:
    evaluation_results: dict - Dizionario che mappa il nome dell'agente ai risultati della valutazione
    save_dir: str - Directory dove salvare i grafici generati
    title_suffix: str - Suffisso da aggiungere al titolo del grafico
    """
    ensure_dir(save_dir)
    
    # Primo grafico: confronto delle ricompense per passo
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, results in evaluation_results.items():
        rewards = results["step_rewards"]
        ax.plot(rewards, label=f"{name} (Total: {results['total_reward']:.2f})")
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Reward')
    ax.set_title(f'Confronto ricompense per passo{title_suffix}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Nome file personalizzato se c'è un suffisso
    filename_base = "post_training_rewards"
    if title_suffix and "Original" in title_suffix:
        filename_base = "post_training_rewards_original"
    elif title_suffix and "Finetuned" in title_suffix:
        filename_base = "post_training_rewards_finetuned"
    
    plt.savefig(f"{save_dir}/{filename_base}.png", bbox_inches='tight', dpi=300)
    plt.show()
    
    # Secondo grafico: distribuzione cumulativa delle ricompense
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, results in evaluation_results.items():
        rewards = results["step_rewards"]
        cumulative_rewards = np.cumsum(rewards)
        ax.plot(cumulative_rewards, label=f"{name}")
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'Ricompense cumulative{title_suffix}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Nome file personalizzato per il cumulativo
    cumulative_filename = "post_training_cumulative_rewards"
    if title_suffix and "Original" in title_suffix:
        cumulative_filename = "post_training_cumulative_rewards_original"
    elif title_suffix and "Finetuned" in title_suffix:
        cumulative_filename = "post_training_cumulative_rewards_finetuned"
    
    plt.savefig(f"{save_dir}/{cumulative_filename}.png", bbox_inches='tight', dpi=300)
    plt.show()

def plot_training_evolution(training_evolution_dict, save_dir=None, filename="training_evolution.png"):
    """
    Plotta l'evoluzione delle reward DURANTE il training
    
    Parametri:
    training_evolution_dict: dict - Dizionario con rewards e timesteps per ogni modello
    save_dir: str - Directory dove salvare il grafico
    filename: str - Nome del file del grafico
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange']
    labels = {
        'noisy': 'Modello con Poco Rumore',
        'more_noisy': 'Modello con Molto Rumore', 
        'noisy_mean': 'Modello con Media Modificata'
    }
    
    for i, (model_name, data) in enumerate(training_evolution_dict.items()):
        if 'rewards' in data and 'timesteps' in data:
            rewards = data['rewards']
            timesteps = data['timesteps']
            
            plt.plot(timesteps, rewards, 
                    color=colors[i % len(colors)], 
                    label=labels.get(model_name, model_name),
                    linewidth=2,
                    marker='o',
                    markersize=4)
    
    plt.title('Evoluzione delle Reward DURANTE il Training', fontsize=16, fontweight='bold')
    plt.xlabel('Timesteps di Training', fontsize=12)
    plt.ylabel('Reward Totale per Episodio di Valutazione', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        ensure_dir(save_dir)
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Grafico di evoluzione del training salvato: {filepath}")
    
    plt.show()
    
def plot_training_and_final_comparison(evaluation_results, training_evolution_dict, save_dir=None):
    """
    Crea un grafico combinato che mostra sia l'evoluzione durante il training
    che le performance finali
    
    Parametri:
    evaluation_results: dict - Risultati finali dei modelli
    training_evolution_dict: dict - Evoluzione durante il training
    save_dir: str - Directory dove salvare i grafici
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Grafico 1: Evoluzione durante il training
    colors = ['blue', 'red', 'green', 'orange']
    labels = {
        'noisy': 'Poco Rumore',
        'more_noisy': 'Molto Rumore', 
        'noisy_mean': 'Media Modificata'
    }
    
    for i, (model_name, data) in enumerate(training_evolution_dict.items()):
        if 'rewards' in data and 'timesteps' in data:
            rewards = data['rewards']
            timesteps = data['timesteps']
            
            ax1.plot(timesteps, rewards, 
                    color=colors[i % len(colors)], 
                    label=labels.get(model_name, model_name),
                    linewidth=2,
                    marker='o',
                    markersize=4)
    
    ax1.set_title('Evoluzione DURANTE il Training', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Timesteps di Training', fontsize=12)
    ax1.set_ylabel('Reward di Valutazione', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Grafico 2: Performance finali
    model_names = list(evaluation_results.keys())
    final_rewards = [evaluation_results[name]['total_reward'] for name in model_names]
    
    bars = ax2.bar(model_names, final_rewards, 
                   color=[colors[i % len(colors)] for i in range(len(model_names))],
                   alpha=0.7)
    
    ax2.set_title('Performance Finali (Ambiente Pulito)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Reward Totale', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Aggiungi valori sopra le barre
    for bar, reward in zip(bars, final_rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{reward:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_dir:
        ensure_dir(save_dir)
        filepath = os.path.join(save_dir, "training_and_final_comparison.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Grafico combinato salvato: {filepath}")
    
    plt.show()

def plot_online_fine_tuning_simple(online_results, save_dir=None, title="Online Fine-tuning Progress"):
    """
    Crea un grafico semplice per mostrare solo l'evoluzione del fine-tuning online
    
    Parametri:
    online_results: dict - Risultati del fine-tuning online da simple_online_learning
    save_dir: str - Directory dove salvare il grafico
    title: str - Titolo del grafico
    """
    plt.figure(figsize=(12, 6))
    
    # Estrai i dati dalle reward
    step_rewards = online_results['step_rewards']
    total_reward = online_results['total_reward']
    
    # Converti rewards in valori numerici se necessario
    try:
        if isinstance(step_rewards[0], (np.ndarray, list)):
            rewards_numeric = [float(np.mean(r)) if hasattr(r, '__len__') else float(r) for r in step_rewards]
        else:
            rewards_numeric = [float(r) for r in step_rewards]
    except (IndexError, TypeError, ValueError):
        rewards_numeric = step_rewards
    
    # Crea gli step temporali
    timesteps = list(range(len(rewards_numeric)))
    
    # Plot principale
    plt.plot(timesteps, rewards_numeric, 
             color='green', 
             linewidth=2, 
             marker='o', 
             markersize=2,
             label=f'Online Fine-tuning (Total: {total_reward:.2f})')
    
    # Calcola e plotta la media mobile per mostrare il trend
    if len(rewards_numeric) > 10:
        window_size = min(50, len(rewards_numeric) // 10)
        moving_avg = []
        for i in range(len(rewards_numeric)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            avg = np.mean(rewards_numeric[start_idx:end_idx])
            moving_avg.append(avg)
        
        plt.plot(timesteps, moving_avg, 
                 color='red', 
                 linewidth=3, 
                 alpha=0.7,
                 label=f'Trend (media mobile)')
    
    # Aggiungi linea orizzontale per la media totale
    avg_reward = np.mean(rewards_numeric)
    plt.axhline(y=avg_reward, 
                color='orange', 
                linestyle='--', 
                alpha=0.8,
                label=f'Media totale: {avg_reward:.3f}')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Passi di Interazione', fontsize=12)
    plt.ylabel('Reward per Passo', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Aggiungi statistiche come testo
    stats_text = f"""Statistiche Fine-tuning Online:
• Reward totale: {total_reward:.2f}
• Reward medio: {avg_reward:.3f}
• Reward min: {min(rewards_numeric):.3f}
• Reward max: {max(rewards_numeric):.3f}
• Passi totali: {len(rewards_numeric)}"""
    
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes, 
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        ensure_dir(save_dir)
        filepath = os.path.join(save_dir, "online_fine_tuning_simple.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Grafico del fine-tuning online salvato: {filepath}")
    
    plt.show()