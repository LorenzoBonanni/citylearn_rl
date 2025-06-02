import os
from citylearn.dynamics import LSTMDynamics
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory
from constants import BATCH_SIZE, LEARNING_STARTS, EPISODES, RANDOM_SEED
from config import *
from adaptive_building import AdaptiveLSTMDynamicsBuilding
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure

def add_noise_to_observations(observations, noise_level=0.15, noise_mean=0.0, noise_type='gaussian'):
    """
    Aggiunge rumore alle osservazioni in base al tipo di rumore specificato.
    
    Parametri:
    observations: np.ndarray - Osservazioni originali
    noise_level: float - Livello di rumore da aggiungere
    noise_mean: float - Media del rumore (solo per rumore gaussiano)
    noise_type: str - Tipo di rumore ('gaussian' o 'uniform')
    
    Returns:
    np.ndarray - Osservazioni con rumore aggiunto
    """    
    noisy_observations = observations.copy()

    for i, observation in enumerate(observations):
        # Non aggiungere rumore a specifiche osservazioni
        mask = np.zeros_like(noisy_observations[i], dtype=bool)
        indices = [0, 1, 2, -1, -2, -3, -4]  # day_type, hour, occupant_count, power_outage, cooling_set_point
        mask[indices] = True

        if noise_type == 'gaussian':
            noise = np.random.normal(loc=noise_mean, scale=noise_level, size=len(observation))
        elif noise_type == 'uniform':
            noise = np.random.uniform(low=-noise_level, high=noise_level, size=len(observation))
        else:
            raise ValueError(f"Tipo di rumore '{noise_type}' non supportato.")
        #print(f"PRE Noise {noise} Observation: {noisy_observations[i]}")
        noisy_observations[i] += noise
        #print(f"SOMMA Noise {noise} Observation: {noisy_observations[i]}")
        # Mantieni le osservazioni originali dove necessario
        noisy_observations[i][mask] = np.array(observations)[i][mask]
        #print(f"POST Noise {noise} Observation: {noisy_observations[i]}")
    
    return noisy_observations

def override_action_value(action_map, action_key, value, hours=None):
    """
    Sovrascrive i valori di una specifica chiave nell'action map con un valore fisso.
    
    Parametri:
    action_map: dict o list - Action map da modificare (dizionario o lista di dizionari)
    action_key: str - Nome dell'azione da modificare (es. 'electrical_storage')
    value: float - Valore da impostare per l'azione
    hours: list o None - Lista di ore da modificare, o None per tutte le ore
    
    Returns:
    action_map - L'action map modificata o originale in caso di errore
    """
    if action_key not in action_map[0]:
        print(f"Errore: '{action_key}' non trovato nell'action map")
        return action_map
        
    # Determina a quali ore applicare il valore
    if hours is None:
        # Se hours non specificato, applica a tutte le ore
        hours_to_modify = list(range(1,25))
    elif isinstance(hours, list):
        # Se hours è una lista, usa quelle ore
        hours_to_modify = hours
    else:
        print(f"Errore: il parametro 'hours' deve essere None o una lista, non {type(hours)}")
        return action_map
        
    # Se abbiamo un singolo dizionario per tutti gli edifici
    if isinstance(action_map, dict):
        for hour in hours_to_modify:
            if hour in action_map[action_key]:
                action_map[action_key][hour] = value
            else:
                print(f"Avviso: ora {hour} non trovata per l'azione '{action_key}'")
    # Se abbiamo una lista di dizionari, uno per edificio
    elif isinstance(action_map, list):
        for i, building_map in enumerate(action_map):
            if action_key in building_map:
                for hour in hours_to_modify:
                    if hour in building_map[action_key]:
                        building_map[action_key][hour] = value
                    else:
                        print(f"Avviso: ora {hour} non trovata per l'edificio {i}, azione '{action_key}'")
            else:
                print(f"Avviso: azione '{action_key}' non trovata per l'edificio {i}")
    else:
        print(f"Errore: l'action_map deve essere un dizionario o una lista, non {type(action_map)}")
        return action_map
    
    return action_map

def test_rbc(env, n=1):
    """
    Esegue una simulazione dell'ambiente CityLearn e restituisce i KPI risultanti.
    
    Parametri:
    env: CityLearnEnv - L'ambiente CityLearn da simulare
    
    Returns:
    tuple - Coppia (env, kpis) con l'ambiente dopo la simulazione e i KPI risultanti
    """
    
    action_map = []
    #for _ in range(len(env.buildings)): #multi_agent
    building_map = {}
    for action in ACTIVE_ACTIONS:
        building_map[action] = {hour: 0 for hour in range(1,25)}
    action_map.append(building_map)

    action_map = override_action_value(action_map,'cooling_or_heating_device', 0.8, hours=[1,2,3,4,5,6,7,8])

    controller = rbc.CustomRBC(env, action_map)
    if n == 1:
        controller = OptimizedRBC(env)

    #print(action_map)

    observations, _ = env.reset()

    while not env.terminated:
        if n == 1:
            actions = controller.predict(observations)
        elif n == 2:
            actions = controller.predict_temperature_bad(observations)
        else:
            actions = controller.predict_multi_obj(observations)
            
        observations, _, _, _, _ = env.step(actions)

    kpis = cp.get_kpis(env)
    return kpis, env

def train_sac(env, sac_model=None, n=1, batch_size=BATCH_SIZE, learning_starts=LEARNING_STARTS, episodes=EPISODES, seed=RANDOM_SEED, time_steps=None, track_rewards=False, eval_freq=1000, deterministic=True):
    if sac_model is None:
        sac_model = SAC(policy='MlpPolicy',
                    env=env,
                    **CUSTOM_AGENT_KWARGS,
                    seed=seed,
                    verbose=1)
    else:
        sac_model.set_env(env)
        if batch_size != sac_model.batch_size:
            sac_model.batch_size = batch_size

    if time_steps is None:
        time_steps = env.unwrapped.time_steps - 1
        
    callback = None
    reward_tracker = None
    
    if n == 2:
        callback = RBCPureCallback(env, verbose=0)
    
    # Setup callback per tracciare reward durante training
    if track_rewards:
        # Crea un ambiente di valutazione pulito (senza rumore)
        eval_env = CityLearnEnv(**ENV_CONFIG)
        eval_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(eval_env))
        
        reward_tracker = SimpleRewardTracker(
            eval_env=eval_env, 
            eval_freq=eval_freq,
            verbose=1
        )
        
        # Combina i callback se necessario
        if callback is not None:
            callback = CallbackList([callback, reward_tracker])
        else:
            callback = reward_tracker

    total_timesteps = episodes * time_steps
    
    env.reset()
    sac_model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False,
        callback=callback,
        progress_bar=True
    )
    # sac_model.set_logger(configure(folder=None, format_strings=['stdout']))
    # observations, _= env.reset()
    # total_reward = 0
    # step = 0
    # step_rewards = []
    # terminated = False
    # truncated = False

    # while not terminated and not truncated:
    #     action, _ = sac_model.predict(observations, deterministic=deterministic)
    #     next_obs, reward, terminated, truncated, info = env.step(action)
    #     step += 1
    #     total_reward += reward
    #     step_rewards.append(reward)
    #     sac_model.replay_buffer.add(observations, next_obs, action, reward, terminated, [info])
    #     if step > batch_size and step > learning_starts:
    #         sac_model.train(gradient_steps=1,batch_size=batch_size)

    
    #     observations = next_obs

    # Restituisci le reward di training se richiesto
    if track_rewards and reward_tracker:
        return env, sac_model, reward_tracker.training_rewards, reward_tracker.timesteps_evaluated
    else:
        return env, sac_model

def evaluate_sac_performance(env, sac_model, episode_name="Model"):
    """
    Esegue un modello SAC allenato e raccoglie le ricompense per ogni passo di tempo.
    
    Parametri:
    env: CityLearnEnv - Ambiente su cui eseguire il modello
    sac_model: SAC - Modello SAC allenato
    episode_name: str - Nome da assegnare all'episodio per l'identificazione
    
    Returns:
    dict - Dizionario contenente le ricompense e le informazioni sulle azioni
    """
    print(f"Valutazione delle prestazioni di {episode_name}")
    observations, _ = env.reset()
    step_rewards = []
    sac_actions_list = []
    total_reward = 0
    
    try:
        # Esegui il modello fino al termine dell'episodio e raccogli le ricompense
        step_count = 0
        max_steps = env.unwrapped.time_steps * 1.1  # Limite di sicurezza per evitare loop infiniti
        
        while not env.unwrapped.terminated and step_count < max_steps:
            actions, _ = sac_model.predict(observations, deterministic=True)
            observations, rewards, _, _, _ = env.step(actions)
            
            # Gestione delle ricompense (potrebbero essere array o tensori)
            if hasattr(rewards, 'shape') and len(rewards.shape) > 0:
                # Se rewards è un array/tensore con più valori, prendiamo la media
                reward_val = float(np.mean(rewards))
            else:
                # Altrimenti usiamo il valore direttamente
                reward_val = float(rewards)
                
            step_rewards.append(rewards)  # Salviamo la ricompensa originale per l'analisi
            sac_actions_list.append(actions)
            total_reward += reward_val
            step_count += 1
            
        if step_count >= max_steps:
            print(f"Attenzione: raggiunto il limite massimo di passi per {episode_name}")
            
    except Exception as e:
        print(f"Errore durante la valutazione di {episode_name}: {e}")
    
    print(f"Ricompensa totale per {episode_name}: {total_reward:.2f}")
    
    return {
        "name": episode_name,
        "total_reward": total_reward,
        "step_rewards": step_rewards,
        "actions": sac_actions_list
    }

def evaluate_sac_performance_robust(env, sac_model, episode_name="Model"):
    """
    Versione robusta della funzione evaluate_sac_performance che gestisce gli errori
    e standardizza i dati di output indipendentemente dal tipo di edificio.
    
    Questa funzione non richiede di patchare gli edifici, ma gestisce internamente 
    le differenze tra tipi di edifici e possibili errori.
    
    Parametri:
    env: CityLearnEnv - Ambiente su cui eseguire il modello
    sac_model: SAC - Modello SAC allenato
    episode_name: str - Nome da assegnare all'episodio per l'identificazione
    
    Returns:
    dict - Dizionario contenente le ricompense e le informazioni sulle azioni
    """
    print(f"Valutazione robusta delle prestazioni di {episode_name}")
    observations, _ = env.reset()
    step_rewards = []
    sac_actions_list = []
    total_reward = 0
    
    try:
        # Esegui il modello fino al termine dell'episodio e raccoglie le ricompense
        step_count = 0
        
        environment_working = True
        
        while environment_working and step_count < env.unwrapped.time_steps:
            try:
                actions, _ = sac_model.predict(observations, deterministic=True)
                try:
                    next_obs, rewards, terminated, truncated, _ = env.step(actions)
                    try:
                        if terminated or truncated or env.unwrapped.terminated:
                            environment_working = False
                    except AttributeError:
                        if terminated or truncated:
                            environment_working = False
                    
                    observations = next_obs
                except Exception as step_error:
                    print(f"Errore durante l'esecuzione del passo nell'ambiente per {episode_name}: {step_error}")
                    # Crea un valore di ricompensa sicuro
                    rewards = np.zeros_like(step_rewards[-1]) if step_rewards else np.array([0.0])
                    environment_working = False
                
                # Standardizza le ricompense (potrebbero essere array o tensori)
                if hasattr(rewards, 'shape') and len(rewards.shape) > 0:
                    # Se rewards è un array/tensore con più valori, prendiamo la media
                    reward_val = float(np.mean(rewards))
                    # Standardizza il formato per l'analisi successiva
                    if rewards.size > 1:
                        step_rewards.append(np.mean(rewards, axis=0) if rewards.ndim > 1 else rewards)
                    else:
                        step_rewards.append(np.array([reward_val]))
                else:
                    # Altrimenti usiamo il valore direttamente
                    reward_val = float(rewards)
                    step_rewards.append(np.array([reward_val]))
                
                sac_actions_list.append(actions)
                total_reward += reward_val
                step_count += 1
                
            except Exception as e:
                print(f"Errore durante la valutazione al passo {step_count} di {episode_name}: {e}")
                # Aggiungi un valore dummy per mantenere la sincronizzazione
                step_rewards.append(np.array([0.0]))
                sac_actions_list.append(np.zeros_like(sac_actions_list[-1]) if sac_actions_list else np.zeros(env.action_space.shape))
                step_count += 1
        
        if step_count >= env.unwrapped.time_steps:
            print(f"Attenzione: raggiunto il limite massimo di passi per {episode_name}")
            
    except Exception as e:
        print(f"Errore generale durante la valutazione di {episode_name}: {e}")
    
    # Converti i risultati in array numpy per garantire compatibilità
    step_rewards_array = np.array(step_rewards)

    print(f"Ricompensa totale per {episode_name}: {total_reward:.2f}")
    
    return {
        "name": episode_name,
        "total_reward": total_reward,
        "step_rewards": step_rewards_array,
        "actions": sac_actions_list
    }

def create_custom_building_env(temperature_offset=0.0, scale_factor=1.0, reward_function=None, custom_model=None, env=ENV_CONFIG, noise_level=0.15, noise_mean=0.0, adaptation_rate=0.05, blend_weight=0.8, window_size=100):
    """
    Crea un ambiente CityLearn con edifici personalizzati che utilizzano CustomLSTMDynamicsBuilding.
    
    Parametri:
    temperature_offset: float - Offset di temperatura da applicare alla predizione
    scale_factor: float - Fattore di scala da applicare alla predizione
    
    Returns:
    CityLearnEnv - Un nuovo ambiente con edifici personalizzati
    """
    # Creiamo prima un ambiente standard per ottenere gli edifici originali
    env = CityLearnEnv(**env)
    
    env_config = {**ENV_CONFIG}
    if reward_function == None:
        env_config['reward_function'] = SolarPenaltyAndComfortReward
    else:
        env_config['reward_function'] = reward_function

    
    # env.observation_names

    # if 'buildings' in env_config:
    #     env_config.pop('buildings')
    
    # env = CityLearnEnv(**env_config, buildings=[])
    
    # Creiamo gli edifici personalizzati basati su quelli originali
    # custom_buildings = []

    # for building in standard_env.buildings:
    #     if isinstance(building, LSTMDynamicsBuilding):
    #         building_args = {
    #             'energy_simulation': building.energy_simulation,
    #             'weather': building.weather,
    #             'observation_metadata': building.observation_metadata,
    #             'action_metadata': building.action_metadata,
    #             'episode_tracker': building.episode_tracker,
    #             'carbon_intensity': building.carbon_intensity,
    #             'pricing': building.pricing,
    #             'dhw_storage': building.dhw_storage,
    #             'cooling_storage': building.cooling_storage,
    #             'heating_storage': building.heating_storage,
    #             'electrical_storage': building.electrical_storage,
    #             'dhw_device': building.dhw_device,
    #             'cooling_device': building.cooling_device,
    #             'heating_device': building.heating_device,
    #             'pv': building.pv,
    #             'name': building.name,
    #             'dynamics': LSTMDynamics(filepath=building.dynamics.filepath,
    #                             input_observation_names=building.dynamics.input_observation_names,
    #                             input_normalization_minimum=building.dynamics.input_normalization_minimum,
    #                             input_normalization_maximum=building.dynamics.input_normalization_maximum,
    #                             hidden_size=building.dynamics.hidden_size,
    #                             num_layers=building.dynamics.num_layers,
    #                             lookback=building.dynamics.lookback,
    #                             input_size=building.dynamics.input_size),
    #             'electric_vehicle_chargers': [] if building.electric_vehicle_chargers is None else building.electric_vehicle_chargers,
    #         }
    #         if custom_model is None:
    #             custom_model = CustomLSTMDynamicsBuilding
    #             custom_building = custom_model(**building_args, temperature_offset=temperature_offset,
    #                                             dynamics_params={'scale_factor': scale_factor})
    #         elif custom_model == NoisyLSTMDynamicsBuilding:
    #             building_noisy_args = {
    #                 'noise_level': noise_level,
    #                 'noise_mean': noise_mean,
    #                 'apply_noise_to': ACTIVE_OBSERVATIONS,
    #                 'noise_type': 'gaussian',
    #                 'seed': 1
    #             }
    #             custom_building = custom_model(**building_args, **building_noisy_args)
    #         elif custom_model == AdaptiveLSTMDynamicsBuilding:
    #             building_adaptive_args = {
    #                 'adaptation_rate': adaptation_rate,
    #                 'blend_weight': blend_weight,
    #                 'window_size': window_size
    #             }
    #             custom_building = custom_model(**building_args, **building_adaptive_args)

    #         custom_buildings.append(custom_building)
    #     else:
    #         # Manteniamo l'edificio originale, ma assicuriamoci che electric_vehicle_chargers sia una lista
    #         if hasattr(building, 'electric_vehicle_chargers') and building.electric_vehicle_chargers is None:
    #             building.electric_vehicle_chargers = []
    #         custom_buildings.append(building)
    
    # env.buildings = custom_buildings

    env.reset()
        
    return env

def simple_online_learning(env, model, update_freq=10, batch_size=10, gradient_steps=4, verbose=1, deterministic=True):
    """
    Vero online fine-tuning: aggiorna il modello ogni pochi passi durante l'interazione
    """        
    model.set_logger(configure(folder=None, format_strings=['stdout']))
    observations, _= env.reset()
    total_reward = 0
    step = 0
    step_rewards = []
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, _ = model.predict(observations, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        total_reward += reward
        step_rewards.append(reward)
        model.replay_buffer.add(observations, next_obs, action, reward, terminated, [info])
        if step % update_freq == 0 and step > 0:
            for _ in range(gradient_steps):
                model.train(gradient_steps=gradient_steps,batch_size=batch_size)
                
            if verbose:
                print(f"Step {step}: Modello aggiornato. Reward totale: {total_reward:.2f}")
    
        observations = next_obs
        
        if terminated or truncated:
            break
    
    return {
        'total_reward': total_reward,
        'step_rewards': step_rewards,
        'final_observations': observations
    }

def evaluate_model_performance(env, model, n_episodes=3, deterministic=True, evaluation_length=None):
    """
    Valuta la performance del modello sull'ambiente.
    
    Args:
        env: L'ambiente di valutazione
        model: Il modello da valutare
        n_episodes: Numero di episodi da eseguire
        deterministic: Se True, usa policy deterministica
        evaluation_length: Lunghezza massima di ogni episodio (se None, esegue l'episodio completo)
        
    Returns:
        mean_reward: La ricompensa media su tutti gli episodi
    """
    eval_env = env
    if evaluation_length is None:
        evaluation_length = env.unwrapped.time_steps - 1

    obs, _ = eval_env.reset()
    episode_reward = 0
    done = False
    
    steps = 0
    while not done and (evaluation_length is None or steps < evaluation_length):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        episode_reward += float(np.mean(reward) if hasattr(reward, 'shape') else reward)
        done = terminated or truncated
        steps += 1
        
    
    return episode_reward

def print_model_params(model, name):
    """Stampa alcuni parametri del modello per verificare le differenze"""
    try:
        # Ottieni i primi 5 parametri del critic network
        critic_params = list(model.critic.parameters())[0].data.flatten()[:5]
        print(f"{name} - Primi 5 parametri critic: {critic_params}")
        
        # Ottieni i primi 5 parametri dell'actor network  
        actor_params = list(model.actor.parameters())[0].data.flatten()[:5]
        print(f"{name} - Primi 5 parametri actor: {actor_params}")
        
    except Exception as e:
        print(f"Errore nel debug del modello {name}: {e}")

def performance_evaluation(first_reward, second_reward, model_name):
    if abs(first_reward) > 0:
        improvement_ft = ((second_reward - first_reward) / abs(first_reward)) * 100
        print(f"\nIl modello {model_name} ha prodotto un cambiamento del {improvement_ft:.2f}% nella performance")

        if improvement_ft > 0:
            print(f"→ Il modello {model_name} ha MIGLIORATO le prestazioni del {abs(improvement_ft):.2f}%")
        else:
            print(f"→ Il modello {model_name} ha PEGGIORATO le prestazioni del {abs(improvement_ft):.2f}%")

class SimpleRewardTracker(BaseCallback):
    """
    Callback semplice per tracciare le reward durante il training
    è una valutazione veloce che si esegue ogni eval_freq timesteps sempre sui primi max_steps.    
    """
    def __init__(self, eval_env, eval_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.training_rewards = []
        self.timesteps_evaluated = []
        
    def _on_step(self) -> bool:
        # Valuta ogni eval_freq timesteps
        if self.n_calls % self.eval_freq == 0:
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            steps = 0
            max_steps = min(self.eval_env.unwrapped.time_steps, 100)  # Valutazione veloce
            
            while not self.eval_env.terminated and steps < max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_reward += float(np.mean(reward) if hasattr(reward, 'shape') else reward)
                steps += 1
                if terminated or truncated:
                    break
                    
            self.training_rewards.append(episode_reward)
            self.timesteps_evaluated.append(self.n_calls)
            
            if self.verbose > 0:
                print(f"Timestep {self.n_calls}: Reward = {episode_reward:.2f}")
        
        return True