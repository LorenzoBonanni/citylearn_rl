import os
from citylearn.dynamics import LSTMDynamics
import numpy as np
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory
from constants import ENV_CONFIG_3
from config import *
from adaptive_building import AdaptiveLSTMDynamicsBuilding

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

def train_sac(env, n=1, episodes=EPISODES):
    sac_model = SAC(policy='MlpPolicy',
                    env=env,
                    **CUSTOM_AGENT_KWARGS,
                    verbose=1)
    if n==1:
        callback=None
    elif n==2:
        callback = RBCPureCallback(env,verbose=0)

    for i in tqdm(range(episodes)):
        sac_model.learn(
            total_timesteps=episodes*(env.unwrapped.time_steps - 1),
            reset_num_timesteps=False,
            callback=callback,
            progress_bar=True
        )

    observations, _ = env.reset()
    sac_actions_list = []

    # Al termine dell'allenamento, esegui il modello per raccogliere le azioni
    while not env.unwrapped.terminated:
        actions, _ = sac_model.predict(observations, deterministic=False)
        observations, _, _, _, _ = env.step(actions)
        sac_actions_list.append(actions)
        
        current_step = env.unwrapped.time_step - 1  
        
        if current_step % 50 == 0 or env.unwrapped.terminated:
            print(f"\n--- Timestep {current_step} ---")
            for b_idx, building in enumerate(env.unwrapped.buildings):
                if hasattr(building, 'indoor_dry_bulb_temperature') and current_step >= 0:
                    try:
                        temp = float(np.ravel(building.indoor_dry_bulb_temperature)[current_step])
                        print(f"Building {b_idx}, Indoor temperature: {temp:.2f}°C")
                    except (IndexError, AttributeError) as e:
                        print(f"Could not retrieve temperature for Building {b_idx}: {e}")
    
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

def create_and_run_adaptive_building_experiment(sac_original):
    """
    Crea e valuta un modello SAC con adaptive buildings, esegue la simulazione e raccoglie le metriche.
    Restituisce sac_adaptive, adaptive_env, adaptation_metrics
    """
    # Crea una seconda copia del modello originale per gli edifici adattivi
    with TemporaryDirectory() as temp_dir:
        adaptive_model_path = os.path.join(temp_dir, "sac_adaptive.zip")
        sac_original.save(adaptive_model_path)
        sac_adaptive = SAC.load(adaptive_model_path)
    adaptive_env = create_custom_building_env(
        custom_model=AdaptiveLSTMDynamicsBuilding,
        adaptation_rate=0.05,
        blend_weight=0.8,
        window_size=100
    )
    if adaptive_env is None:
        print("ERRORE: Non è stato possibile creare l'ambiente con edifici adattivi.")
        return sac_adaptive, None, None
    adaptive_env = StableBaselines3Wrapper(NormalizedObservationWrapper(adaptive_env))
    observations, _ = adaptive_env.reset()
    obs_shape = observations.shape[0] if hasattr(observations, 'shape') else len(observations)
    expected_shape = sac_adaptive.observation_space.shape[0]
    if obs_shape > expected_shape:
        if isinstance(observations, np.ndarray):
            observations = observations[:expected_shape]
        elif isinstance(observations, (list, tuple)):
            observations = type(observations)(list(observations)[:expected_shape])
    elif obs_shape < expected_shape:
        if isinstance(observations, np.ndarray):
            pad_width = expected_shape - obs_shape
            observations = np.concatenate([observations, np.zeros(pad_width, dtype=observations.dtype)])
        elif isinstance(observations, list):
            observations = observations + [0.0] * (expected_shape - obs_shape)
        elif isinstance(observations, tuple):
            observations = tuple(list(observations) + [0.0] * (expected_shape - obs_shape))
        else:
            observations = [0.0] * expected_shape
    total_steps = min(2000, adaptive_env.unwrapped.time_steps-1)
    adaptation_metrics = []
    for step in range(total_steps):
        obs_shape = observations.shape[0] if hasattr(observations, 'shape') else len(observations)
        expected_shape = sac_adaptive.observation_space.shape[0]
        if obs_shape > expected_shape:
            if isinstance(observations, np.ndarray):
                observations = observations[:expected_shape]
            elif isinstance(observations, (list, tuple)):
                observations = type(observations)(list(observations)[:expected_shape])
        elif obs_shape < expected_shape:
            if isinstance(observations, np.ndarray):
                pad_width = expected_shape - obs_shape
                observations = np.concatenate([observations, np.zeros(pad_width, dtype=observations.dtype)])
            elif isinstance(observations, list):
                observations = observations + [0.0] * (expected_shape - obs_shape)
            elif isinstance(observations, tuple):
                observations = tuple(list(observations) + [0.0] * (expected_shape - obs_shape))
            else:
                observations = [0.0] * expected_shape
        action, _ = sac_adaptive.predict(observations, deterministic=True)
        action_shape = action.shape[0] if hasattr(action, 'shape') else len(action)
        expected_action_shape = adaptive_env.action_space.shape[0]
        if action_shape > expected_action_shape:
            if isinstance(action, np.ndarray):
                action = action[:expected_action_shape]
            elif isinstance(action, (list, tuple)):
                action = type(action)(list(action)[:expected_action_shape])
        elif action_shape < expected_action_shape:
            if isinstance(action, np.ndarray):
                pad_width = expected_action_shape - action_shape
                action = np.concatenate([action, np.zeros(pad_width, dtype=action.dtype)])
            elif isinstance(action, list):
                action = action + [0.0] * (expected_action_shape - action_shape)
            elif isinstance(action, tuple):
                action = tuple(list(action) + [0.0] * (expected_action_shape - action_shape))
            else:
                action = [0.0] * expected_action_shape
        observations, rewards, terminated, truncated, info = adaptive_env.step(action)
        if step % 100 == 0:
            metrics = {}
            try:
                for i, building in enumerate(adaptive_env.unwrapped.buildings):
                    if hasattr(building, 'get_adaptation_metrics'):
                        building_metrics = building.get_adaptation_metrics()
                        metrics[f"building_{i+1}"] = building_metrics
                adaptation_metrics.append((step, metrics))
            except Exception as e:
                print(f"Errore nella raccolta delle metriche di adattamento: {e}")
        if terminated or truncated:
            break
    return sac_adaptive, adaptive_env, adaptation_metrics

def evaluate_adaptive_building_model(sac_adaptive):
    """
    Valuta il modello SAC con adaptive buildings su un ambiente di test e restituisce le metriche finali.
    """
    adaptive_eval_env = create_adaptive_building_env(
        ENV_CONFIG_3.copy(),
        adaptation_rate=0.05,
        blend_weight=0.8,
        window_size=100
    )
    if adaptive_eval_env is None:
        print("ERRORE: Non è stato possibile creare l'ambiente adattivo per la valutazione finale.")
        return {
            "name": "SAC-Adaptive-Building",
            "total_reward": float('nan'),
            "step_rewards": np.array([]),
            "actions": []
        }, None
    adaptive_eval_env = StableBaselines3Wrapper(NormalizedObservationWrapper(adaptive_eval_env))
    final_adaptive_eval = evaluate_sac_performance_robust(adaptive_eval_env, sac_adaptive, "SAC-Adaptive-Building")
    final_adaptation_metrics = {}
    try:
        for i, building in enumerate(adaptive_eval_env.unwrapped.buildings):
            if hasattr(building, 'get_adaptation_metrics'):
                metrics = building.get_adaptation_metrics()
                final_adaptation_metrics[f"building_{i+1}"] = metrics
    except Exception as e:
        print(f"Errore nell'estrazione delle metriche di adattamento finali: {e}")
    return final_adaptive_eval, final_adaptation_metrics

def create_adaptive_building_env(config_dict, adaptation_rate=0.05, blend_weight=0.8, window_size=100):
    """
    Crea un ambiente CityLearn con edifici adattivi che aggiornano il loro modello di transizione.
    Args:
        config_dict: Dizionario di configurazione dell'ambiente
        adaptation_rate: Tasso di adattamento del modello
        blend_weight: Peso iniziale del modello originale (1.0 = solo originale, 0.0 = solo nuovo)
        window_size: Dimensione della finestra di osservazioni per l'apprendimento
    Returns:
        CityLearnEnv: Ambiente con edifici adattivi
    """
    from citylearn.citylearn import CityLearnEnv
    from adaptive_building import AdaptiveLSTMDynamicsBuilding
    # Estrai parametri di configurazione
    schema_name = config_dict.get('schema')
    building_ids = config_dict.get('buildings')
    simulation_start_time_step = config_dict.get('simulation_start_time_step', 0)
    simulation_end_time_step = config_dict.get('simulation_end_time_step', 8760)
    active_observations = config_dict.get('active_observations', [])
    central_agent = config_dict.get('central_agent', False)
    try:
        temp_env = CityLearnEnv(**config_dict)
        adaptive_buildings = []
        for i, building_id in enumerate(building_ids):
            try:
                original_building = temp_env.buildings[i]
                if hasattr(original_building, 'dynamics') and original_building.dynamics is not None:
                    orig_dynamics = original_building.dynamics
                    adaptive_building = AdaptiveLSTMDynamicsBuilding(
                        buildingId=original_building.name,
                        weather=original_building.weather,
                        dhw_storage=original_building.dhw_storage,
                        cooling_storage=original_building.cooling_storage,
                        dynamics=orig_dynamics,
                        adaptation_rate=adaptation_rate,
                        blend_weight=blend_weight,
                        window_size=window_size,
                        observation_metadata=original_building.observation_metadata,
                        action_metadata=original_building.action_metadata,
                        carbon_intensity=original_building.carbon_intensity,
                        pricing=original_building.pricing,
                        seconds_per_time_step=original_building.seconds_per_time_step,
                        random_seed=original_building.random_seed,
                        episode_tracker=original_building.episode_tracker,
                        simulate_power_outage=original_building.simulate_power_outage
                    )
                    adaptive_building.observation_space = original_building.observation_space
                    adaptive_building.action_space = original_building.action_space
                    adaptive_buildings.append(adaptive_building)
                else:
                    return None
            except Exception:
                return None
        if not adaptive_buildings:
            return None
        env = CityLearnEnv(
            buildings=adaptive_buildings,
            schema=schema_name,
            central_agent=central_agent,
            simulation_start_time_step=simulation_start_time_step,
            simulation_end_time_step=simulation_end_time_step,
            active_observations=active_observations if active_observations else None
        )
        return env
    except Exception:
        return None

def simple_online_learning(env, model, n_steps=5000, update_freq=100, batch_size=64, 
                           gradient_steps=10, verbose=1, episodes=EPISODES):
    """
    Esegue un ciclo di apprendimento online semplice con aggiornamento periodico del modello.
    
    Args:
        env: L'ambiente di interazione
        model: Il modello RL pre-addestrato (es. SAC)
        n_steps: Numero totale di passi di interazione PER EPISODIO
        update_freq: Frequenza di aggiornamento del modello
        batch_size: Dimensione del batch per l'aggiornamento
        gradient_steps: Numero di passi di gradiente ad ogni aggiornamento
        verbose: Livello di verbosità
        episodes: Numero di episodi di training
        
    Returns:
        performance_history: Lista di tuple (step, reward) che tracciano la performance
    """
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    
    # Configura il buffer di esperienza con le dimensioni corrette
    if hasattr(model, 'replay_buffer'):
        try:
            from stable_baselines3.common.buffers import ReplayBuffer
            
            # Ottieni dimensioni dallo spazio di osservazione e azione
            obs_dim = env.observation_space.shape[0] if len(env.observation_space.shape) > 0 else 1
            action_dim = env.action_space.shape[0] if len(env.action_space.shape) > 0 else 1
            
            # Crea un nuovo buffer con le dimensioni corrette
            new_buffer = ReplayBuffer(
                buffer_size=model.replay_buffer.buffer_size,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=model.device,
                n_envs=1,
                optimize_memory_usage=model.replay_buffer.optimize_memory_usage
            )
            
            # Sostituisci il buffer esistente
            model.replay_buffer = new_buffer
            print("Buffer di replay ricreato con le dimensioni corrette.")
        except Exception as e:
            print(f"Impossibile ricreare il buffer di replay: {e}")
    
    performance_history = []
    total_steps = episodes * n_steps
    
    # Barra di progresso principale per tutti gli episodi
    pbar = tqdm(total=total_steps, desc="Online Learning")
    
    global_step = 0
    total_reward_all_episodes = 0
    
    # Ciclo principale per episodi
    for episode in range(episodes):
        # Reset dell'ambiente per nuovo episodio
        observations, _ = env.reset()
        episode_reward = 0
        
        # Ciclo per steps all'interno dell'episodio
        for step in range(1, n_steps + 1):
            try:
                global_step += 1
                
                # Predici l'azione (usa deterministic=False per esplorare durante l'apprendimento)
                action, _ = model.predict(observations, deterministic=True)
                
                # Esegui l'azione
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Calcola reward totale per questo passo
                step_reward = float(np.mean(reward) if hasattr(reward, 'shape') else reward)
                episode_reward += step_reward
                total_reward_all_episodes += step_reward
                
                # Aggiungi l'esperienza al buffer di replay
                if hasattr(model, 'replay_buffer'):
                    model.replay_buffer.add(
                        observations,
                        next_obs,
                        action,
                        reward,
                        terminated or truncated,
                        [info]
                    )
                
                # Aggiorna il modello periodicamente
                if global_step % update_freq == 0 and global_step > batch_size:
                    if verbose > 0:
                        avg_reward = total_reward_all_episodes / global_step
                        pbar.set_description(f"Ep {episode+1}/{episodes} - Avg Reward: {avg_reward:.2f}")
                    
                    model.train(gradient_steps=gradient_steps, batch_size=batch_size)
            
                observations = next_obs
                pbar.update(1)
                
                # Se l'episodio termina naturalmente, interrompi il loop interno
                if terminated or truncated:
                    if verbose > 0:
                        pbar.set_description(f"Ep {episode+1}/{episodes} completed - Ep Reward: {episode_reward:.2f}")
                    break
                    
            except Exception as e:
                print(f"Errore all'episodio {episode+1}, passo {step}: {e}")
                continue
        
        # Traccia le performance alla fine di ogni episodio
        performance_history.append((global_step, episode_reward))
        
        if verbose > 0:
            print(f"\nEpisodio {episode+1}/{episodes} completato - Reward: {episode_reward:.2f}")
    
    pbar.close()
    
    print(f"\nApprendimento online completato:")
    print(f"- Episodi: {episodes}")
    print(f"- Passi totali: {global_step}")
    print(f"- Reward medio per episodio: {total_reward_all_episodes/episodes:.2f}")
    
    return performance_history

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
    # Crea un nuovo ambiente per la valutazione
    eval_env = env
    if evaluation_length is None:
        evaluation_length = min((env.unwrapped.time_steps - 1)/4,200)

    # Esegui n_episodes e calcola la ricompensa media
    episode_rewards = []
    for _ in range(n_episodes):
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
            
        episode_rewards.append(episode_reward)
    
    # Calcola e restituisci la ricompensa media
    return sum(episode_rewards) / n_episodes

# Dopo l'allenamento di ogni modello, stampa alcuni parametri
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