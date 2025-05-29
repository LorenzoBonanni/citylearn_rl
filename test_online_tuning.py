from functools import partial
from custom_function import *
import os
import numpy as np
from gymnasium.wrappers import TransformObservation
from adaptive_building import AdaptiveLSTMDynamicsBuilding
from tempfile import TemporaryDirectory
from stable_baselines3 import SAC


def add_noise_to_observations(observations, noise_level=0.15, noise_mean=0.0, noise_type='gaussian'):
    """
    Aggiunge rumore alle osservazioni in base al tipo di rumore specificato.
    
    Parametri:
    observations: np.ndarray - Osservazioni originali
    noise_level: float - Livello di rumore da aggiungere
    noise_mean: float - Media del rumore (solo per rumore gaussiano)
    noise_type: str - Tipo di rumore ('gaussian' o 'uniform')
    seed: int - Semenza per la riproducibilità
    Returns:
    np.ndarray - Osservazioni con rumore aggiunto
    """    
    noisy_observations = observations.copy()

    for i, observation in enumerate(observations):
        # non aggiungre rumoore a 'day_type', 'hour', 'occupant_count', 'power_outage', 'indoor_dry_bulb_temperature_cooling_set_point'
        mask = np.zeros_like(noisy_observations[i], dtype=bool)
        # active_observations: ['day_type', 'hour', 'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_1', 'outdoor_dry_bulb_temperature_predicted_2', 'outdoor_dry_bulb_temperature_predicted_3', 'diffuse_solar_irradiance', 'diffuse_solar_irradiance_predicted_1', 'diffuse_solar_irradiance_predicted_2', 'diffuse_solar_irradiance_predicted_3', 'direct_solar_irradiance', 'direct_solar_irradiance_predicted_1', 'direct_solar_irradiance_predicted_2', 'direct_solar_irradiance_predicted_3', 'carbon_intensity', 'indoor_dry_bulb_temperature', 'non_shiftable_load', 'solar_generation', 'dhw_storage_soc', 'electrical_storage_soc', 'net_electricity_consumption', 'electricity_pricing', 'electricity_pricing_predicted_1', 'electricity_pricing_predicted_2', 'electricity_pricing_predicted_3', 'cooling_demand', 'dhw_demand', 'occupant_count', 'power_outage', 'indoor_dry_bulb_temperature_cooling_set_point']
        indices = [0, 1, -1, -2, -3]
        mask[indices] = True  # Indici delle osservazioni da non modificare

        if noise_type == 'gaussian':
            noise = np.random.normal(loc=noise_mean, scale=noise_level, size=len(observation))
        elif noise_type == 'uniform':
            noise = np.random.uniform(low=-noise_level, high=noise_level, size=len(observation))
        else:
            raise ValueError(f"Tipo di rumore '{noise_type}' non supportato.")
        noisy_observations[i] += noise
        # Mantieni le osservazioni originali dove necessario
        noisy_observations[i][mask] = np.array(observations)[i][mask] # Mantieni le osservazioni originali dove necessario
    return noisy_observations

def train_and_transfer_sac():
    """
    Allenamento iniziale, trasferimento e fine-tuning continuo
    di un modello SAC tra ambienti diversi.
    """
    print("Fase 1: Addestramento iniziale su ambiente standard")

    # Ambiente di training iniziale
    train_env = CityLearnEnv(**ENV_CONFIG)
    train_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(TransformObservation(train_env, partial(add_noise_to_observations, noise_type='gaussian', noise_level=0.15, noise_mean=0.0))))
    # Addestramento SAC iniziale
    _, sac_model = train_sac(train_env)
    # Crea una vera copia del modello originale usando save e load
    # Crea una directory temporanea
    with TemporaryDirectory() as temp_dir:
        # Salva il modello originale
        original_model_path = os.path.join(temp_dir, "sac_original.zip")
        sac_model.save(original_model_path)
        
        # Carica una vera copia del modello
        sac_original = SAC.load(original_model_path)

    print("Fase 2: Test e fine-tuning online su ambiente reale (diverso)")
    
    # Ambiente di test/applicazione (ambiente "reale" diverso)
    test_env = CityLearnEnv(**ENV_CONFIG)  
    test_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(test_env))

    print(f"Fine-tuning online per {test_env.unwrapped.time_steps-1} passi temporali")
    
    performance_history = simple_online_learning(
        env=test_env,
        model=sac_model,
        n_steps=test_env.unwrapped.time_steps-1,
        update_freq=UPDATE_FREQ,  # Aggiorna il modello ogni update_freq passi
        batch_size=64,
        gradient_steps=10,        # Numero di aggiornamenti del modello per ogni update_freq passi
        verbose=1,
    )
    
    #ambiente con molto rumore
    print("Fase 5: Test di un modello SAC con rumore")
    train_env = CityLearnEnv(**ENV_CONFIG)
    train_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(TransformObservation(train_env, partial(add_noise_to_observations, noise_type='gaussian', noise_level=0.45, noise_mean=0.0))))
    _, sac_noisy = train_sac(train_env)

    #ambiente con media modificata
    print("Fase 6: Test di un modello SAC con media modificata")
    train_env = CityLearnEnv(**ENV_CONFIG)
    train_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(TransformObservation(train_env, partial(add_noise_to_observations, noise_type='gaussian', noise_level=0.15, noise_mean=1.0))))
    _, sac_noisy_mean = train_sac(train_env)

    # Ambiente di valutazione standard
    eval_env = CityLearnEnv(**ENV_CONFIG)
    eval_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(eval_env))
    
    # Valuta i 4 modelli
    print("\nValutazione del modello originale senza adattamenti:")
    final_original_eval = evaluate_sac_performance_robust(eval_env, sac_original, "SAC-Noisy")
    
    print("\nValutazione del modello con fine-tuning tradizionale:")
    final_real_eval = evaluate_sac_performance_robust(eval_env, sac_model, "SAC-Noisy-Finetuned")
    
    print("\nValutazione del modello con rumore:")
    final_noisy_eval = evaluate_sac_performance_robust(eval_env, sac_noisy, "SAC-High-Noisy")

    print("\nValutazione del modello con media modificata:")
    final_noisy_mean_eval = evaluate_sac_performance_robust(eval_env, sac_noisy_mean, "SAC-Noisy-Mean")
    
    # restituisco performance dei diversi ambienti
    results = {
        "real_env": final_real_eval,
        "original_env": final_original_eval,
        "noisy_env": final_noisy_eval,
        "noisy_mean_env": final_noisy_mean_eval,
    }

    print_model_params(sac_original, "SAC-Original")
    print_model_params(sac_model, "SAC-Fine-tuned") 
    print_model_params(sac_noisy, "SAC-Noisy")
    print_model_params(sac_noisy_mean, "SAC-Noisy-Mean")
    
    return results

if __name__ == "__main__":
    import time
    from datetime import datetime
    
    # Assicurati che la directory per i plot esista
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Crea una directory specifica per questa esecuzione
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(SAVE_DIR, f"adaptive_comparison_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"I risultati saranno salvati in: {run_dir}")
    
    start_time = time.time()
    try:

        evaluation = train_and_transfer_sac()
        print("\n========== RISULTATI FINALI ==========")
        
        if 'original_env' in evaluation and 'real_env' in evaluation and 'noisy_env' in evaluation and 'noisy_mean_env' in evaluation and 'adaptive_env' in evaluation:
            print(f"Ricompensa nell'ambiente originale: {evaluation['original_env']['total_reward']:.2f}")
            print(f"Ricompensa nell'ambiente con fine-tuning: {evaluation['real_env']['total_reward']:.2f}")
            print(f"Ricompensa nell'ambiente con rumore: {evaluation['noisy_env']['total_reward']:.2f}")
            print(f"Ricompensa nell'ambiente con media modificata: {evaluation['noisy_mean_env']['total_reward']:.2f}")
            print(f"Ricompensa nell'ambiente con edifici adattivi: {evaluation['adaptive_env']['total_reward']:.2f}")
        
            # Calcolo del miglioramento percentuale
            real_reward = evaluation['real_env']['total_reward']
            orig_reward = evaluation['original_env']['total_reward']
            noisy_reward = evaluation['noisy_env']['total_reward']
            noisy_mean_reward = evaluation['noisy_mean_env']['total_reward']
            adaptive_reward = evaluation['adaptive_env']['total_reward']


            performance_evaluation(orig_reward, real_reward, "fine-tuning")
            performance_evaluation(orig_reward, noisy_reward, "con rumore")
            performance_evaluation(orig_reward, noisy_mean_reward, "con media modificata")
            performance_evaluation(orig_reward, adaptive_reward, "con edifici adattivi")

        # Prepara i risultati nel formato atteso da plot_post_training_rewards
        evaluation_results = {}
        
        if 'original_env' in evaluation:
            evaluation_results['SAC-Originale'] = {
                'total_reward': evaluation['original_env']['total_reward'],
                'step_rewards': evaluation['original_env']['step_rewards']
            }
            
        if 'real_env' in evaluation:
            evaluation_results['SAC-Fine-tuned'] = {
                'total_reward': evaluation['real_env']['total_reward'],
                'step_rewards': evaluation['real_env']['step_rewards']
            }
        if 'noisy_env' in evaluation:
            evaluation_results['SAC-Noisy'] = {
                'total_reward': evaluation['noisy_env']['total_reward'],
                'step_rewards': evaluation['noisy_env']['step_rewards']
            }
        if 'noisy_mean_env' in evaluation:
            evaluation_results['SAC-Noisy-Mean'] = {
                'total_reward': evaluation['noisy_mean_env']['total_reward'],
                'step_rewards': evaluation['noisy_mean_env']['step_rewards']
            }
        if 'adaptive_env' in evaluation:            
            evaluation_results['SAC-Adattivo'] = {
                'total_reward': evaluation['adaptive_env']['total_reward'],
                'step_rewards': evaluation['adaptive_env']['step_rewards']
            }
        
        # Plot delle ricompense dopo il training
        try:
            import custom_plot as cp
            cp.plot_post_training_rewards(
                evaluation_results,
                save_dir=run_dir,
            )
            print(f"\nI grafici di confronto sono stati salvati in: {run_dir}")
        except Exception as e:
            print(f"Errore nella generazione dei grafici: {e}")
            
        # Salva un riepilogo dei risultati in un file di testo
        try:
            with open(os.path.join(run_dir, "summary_results.txt"), "w") as f:
                f.write("===== RISULTATI ESPERIMENTO ADAPTIVE BUILDINGS =====\n\n")
                f.write(f"Data esperimento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Ricompense totali:\n")
                f.write(f"- Modello originale: {evaluation['original_env']['total_reward']:.2f}\n")
                f.write(f"- Modello fine-tuned: {evaluation['real_env']['total_reward']:.2f}\n")
                f.write(f"- Modello con rumore: {evaluation['noisy_env']['total_reward']:.2f}\n")
                f.write(f"- Modello con media modificata: {evaluation['noisy_mean_env']['total_reward']:.2f}\n")
                f.write(f"- Modello con edifici adattivi: {evaluation['adaptive_env']['total_reward']:.2f}\n\n")
        except Exception as e:
            print(f"Errore nel salvataggio del riepilogo: {e}")
            
    except Exception as e:
        print(f"\nERRORE durante l'esecuzione dell'esperimento: {e}")
        import traceback
        traceback.print_exc()
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nEsperimento completato in {duration/60:.2f} minuti ({duration:.1f} secondi)")