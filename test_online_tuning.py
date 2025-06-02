from functools import partial
from custom_function import *
import os
import numpy as np
from gymnasium.wrappers import TransformObservation
from adaptive_building import AdaptiveLSTMDynamicsBuilding
from tempfile import TemporaryDirectory
from stable_baselines3 import SAC

def train_and_transfer_sac():
    """
    Allenamento iniziale, trasferimento e fine-tuning continuo
    di un modello SAC tra ambienti diversi.
    """
    print("Fase 1: Addestramento iniziale su ambiente standard")

    normal_env = CityLearnEnv(**ENV_CONFIG)
    normal_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(normal_env))
    _, sac_normal, training_rewards_normal, timesteps_normal = train_sac(
        normal_env, 
        track_rewards=True,
        eval_freq=50  
    )
    print("Fase 2: Addestramento su ambiente con rumore")
    noisy_env = CityLearnEnv(**ENV_CONFIG)
    noisy_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(TransformObservation(noisy_env, partial(add_noise_to_observations, noise_type='gaussian', noise_level=0.45, noise_mean=0.0))))
    _, sac_noisy, training_rewards_noisy, timesteps_noisy = train_sac(
        noisy_env, 
        track_rewards=True, 
        eval_freq=50  
    )
    sac_fine_tuned = None
    with TemporaryDirectory() as temp_dir:
        # Salva il modello originale
        original_model_path = os.path.join(temp_dir, "sac_original.zip")
        sac_noisy.save(original_model_path)
        
        # Carica una vera copia del modello
        sac_fine_tuned = SAC.load(original_model_path)

    print("Fase 3: Test e fine-tuning online su ambiente reale (diverso)")

    
    sac_fine_tuned.set_env(normal_env)

    # sac_fine_tuned.learn(
    #         total_timesteps=normal_env.unwrapped.time_steps-1,
    #         reset_num_timesteps=False,
    #         progress_bar=True
    #     )

    online_results = simple_online_learning(
        env=normal_env,
        model=sac_fine_tuned,
        update_freq=4, 
        batch_size=8,    
        gradient_steps=4, 
        verbose=1,
    )
    
    # Estrai i dati corretti dal dizionario restituito
    training_rewards_fine = online_results['step_rewards']
    # Crea timesteps fittizi per il fine-tuning online (non abbiamo timesteps reali)
    timesteps_fine = list(range(0, len(training_rewards_fine) * 4, 4))  # Ogni 4 step
    
    #ambiente con molto rumore
    print("Fase 4: Test di un modello SAC con rumore")
    more_noise_env = CityLearnEnv(**ENV_CONFIG)
    more_noise_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(TransformObservation(more_noise_env, partial(add_noise_to_observations, noise_type='gaussian', noise_level=1, noise_mean=0.0))))
    _, sac_more_noise, training_rewards_more_noise, timesteps_more_noise = train_sac(
        more_noise_env,
        track_rewards=True,
        eval_freq=50
    )

    #ambiente con media modificata
    print("Fase 5: Test di un modello SAC con media modificata")
    noisy_mean_env = CityLearnEnv(**ENV_CONFIG)
    noisy_mean_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(TransformObservation(noisy_mean_env, partial(add_noise_to_observations, noise_type='gaussian', noise_level=0.15, noise_mean=1.0))))
    _, sac_noisy_mean, training_rewards_noisy_mean, timesteps_noisy_mean = train_sac(
        noisy_mean_env,
        track_rewards=True,
        eval_freq=50
    )
    
    print("\nValutazione del modello originale nell'ambiente standard:")
    final_normal = evaluate_sac_performance(normal_env, sac_normal, "SAC-Normal")

    print("\nValutazione del modello con poco rumore:")
    final_noisy = evaluate_sac_performance(normal_env, sac_noisy, "SAC-Noisy")
    
    print("\nValutazione del modello con fine-tuning:")
    final_fine_tuned = evaluate_sac_performance(normal_env, sac_fine_tuned, "SAC-Noisy-Finetuned")
    
    print("\nValutazione del modello con rumore:")
    final_more_noisy = evaluate_sac_performance(normal_env, sac_more_noise, "SAC-High-Noisy")

    print("\nValutazione del modello con media modificata:")
    final_noisy_mean = evaluate_sac_performance(normal_env, sac_noisy_mean, "SAC-Noisy-Mean")
    
    # restituisco performance dei diversi ambienti
    results = {
        "normal_env": final_normal,
        "noisy_env": final_noisy,
        "fine_tuned_env": final_fine_tuned,
        "more_noisy_env": final_more_noisy,
        "noisy_mean_env": final_noisy_mean,
        "training_evolution": {
            "normal": {"rewards": training_rewards_normal, "timesteps": timesteps_normal},
            "noisy": {"rewards": training_rewards_noisy, "timesteps": timesteps_noisy},
            "fine_tuned": {"rewards": training_rewards_fine, "timesteps": timesteps_fine},
            "more_noisy": {"rewards": training_rewards_more_noise, "timesteps": timesteps_more_noise},
            "noisy_mean": {"rewards": training_rewards_noisy_mean, "timesteps": timesteps_noisy_mean}
        }
    }

    print_model_params(sac_noisy, "Noisy")
    print_model_params(sac_fine_tuned, "Fine-tuned") 
    print_model_params(sac_noisy, "More-Noisy")
    print_model_params(sac_noisy_mean, "Noisy-Mean")
    
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
        
        
        if 'noisy_env' in evaluation and 'fine_tuned_env' in evaluation and 'more_noisy_env' in evaluation and 'noisy_mean_env' in evaluation:
            print(f"Ricompensa nell'ambiente normale (baseline): {evaluation['normal_env']['total_reward']:.2f}")
            
            print(f"Ricompensa nell'ambiente noisy: {evaluation['noisy_env']['total_reward']:.2f}")
            print(f"Ricompensa nell'ambiente con fine-tuning: {evaluation['fine_tuned_env']['total_reward']:.2f}")
            print(f"Ricompensa nell'ambiente con molto rumore: {evaluation['more_noisy_env']['total_reward']:.2f}")
            print(f"Ricompensa nell'ambiente con media modificata: {evaluation['noisy_mean_env']['total_reward']:.2f}")
        
            # Calcolo del miglioramento percentuale - usa il modello normale come baseline se disponibile
            baseline_reward = evaluation['normal_env']['total_reward'] if 'normal_env' in evaluation else evaluation['noisy_env']['total_reward']
            real_reward = evaluation['fine_tuned_env']['total_reward']
            noisy_reward = evaluation['more_noisy_env']['total_reward']
            noisy_mean_reward = evaluation['noisy_mean_env']['total_reward']

            # Confronti rispetto al baseline (modello normale)
            if 'normal_env' in evaluation:
                print(f"\n--- CONFRONTI RISPETTO AL MODELLO NORMALE (BASELINE) ---")
                performance_evaluation(baseline_reward, evaluation['noisy_env']['total_reward'], "addestrato con rumore")
                performance_evaluation(baseline_reward, real_reward, "fine-tuning online")
                performance_evaluation(baseline_reward, noisy_reward, "addestrato con molto rumore")
                performance_evaluation(baseline_reward, noisy_mean_reward, "addestrato con media modificata")
            
            # Confronti rispetto al modello con rumore (come prima)
            print(f"\n--- CONFRONTI RISPETTO AL MODELLO CON RUMORE ---")
            performance_evaluation(evaluation['noisy_env']['total_reward'], real_reward, "fine-tuning")
            performance_evaluation(evaluation['noisy_env']['total_reward'], noisy_reward, "con rumore")
            performance_evaluation(evaluation['noisy_env']['total_reward'], noisy_mean_reward, "con media modificata")

        # Prepara i risultati nel formato atteso da plot_post_training_rewards
        evaluation_results = {}
        
        # Aggiungi il modello normale se presente
        if 'normal_env' in evaluation:
            evaluation_results['Normal'] = {
                'total_reward': evaluation['normal_env']['total_reward'],
                'step_rewards': evaluation['normal_env']['step_rewards']
            }
        
        if 'noisy_env' in evaluation:
            evaluation_results['Noisy'] = {
                'total_reward': evaluation['noisy_env']['total_reward'],
                'step_rewards': evaluation['noisy_env']['step_rewards']
            }
            
        if 'fine_tuned_env' in evaluation:
            evaluation_results['Fine-tuned'] = {
                'total_reward': evaluation['fine_tuned_env']['total_reward'],
                'step_rewards': evaluation['fine_tuned_env']['step_rewards']
            }
        if 'more_noisy_env' in evaluation:
            evaluation_results['More-Noise'] = {
                'total_reward': evaluation['more_noisy_env']['total_reward'],
                'step_rewards': evaluation['more_noisy_env']['step_rewards']
            }
        if 'noisy_mean_env' in evaluation:
            evaluation_results['Mean-Noise'] = {
                'total_reward': evaluation['noisy_mean_env']['total_reward'],
                'step_rewards': evaluation['noisy_mean_env']['step_rewards']
            }
      
        cp.plot_post_training_rewards(
            evaluation_results,
            save_dir=run_dir,
        )
        
        # Aggiungi il grafico semplice del fine-tuning online
        if 'fine_tuned_env' in evaluation:
            # Ottieni i risultati del fine-tuning online
            online_results = {
                'step_rewards': evaluation['fine_tuned_env']['step_rewards'],
                'total_reward': evaluation['fine_tuned_env']['total_reward']
            }
            cp.plot_online_fine_tuning_simple(
                online_results,
                save_dir=run_dir,
                title="Evoluzione Fine-tuning Online su Ambiente Pulito"
            )
        
        try:
            with open(os.path.join(run_dir, "summary_results.txt"), "w") as f:
                f.write("===== RISULTATI ESPERIMENTO ADAPTIVE BUILDINGS =====\n\n")
                f.write(f"Data esperimento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Ricompense totali:\n")
                if 'normal_env' in evaluation:
                    f.write(f"- Modello normale (baseline): {evaluation['normal_env']['total_reward']:.2f}\n")
                f.write(f"- Modello con rumore: {evaluation['noisy_env']['total_reward']:.2f}\n")
                f.write(f"- Modello fine-tuned: {evaluation['fine_tuned_env']['total_reward']:.2f}\n")
                f.write(f"- Modello con molto rumore: {evaluation['more_noisy_env']['total_reward']:.2f}\n")
                f.write(f"- Modello con media modificata: {evaluation['noisy_mean_env']['total_reward']:.2f}\n")
        except Exception as e:
            print(f"Errore nel salvataggio del riepilogo: {e}")
            
    except Exception as e:
        print(f"\nERRORE durante l'esecuzione dell'esperimento: {e}")
        import traceback
        traceback.print_exc()
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nEsperimento completato in {duration/60:.2f} minuti ({duration:.1f} secondi)")