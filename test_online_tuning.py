from custom_function import *
from stable_baselines3.common.callbacks import BaseCallback
import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from adaptive_building import AdaptiveLSTMDynamicsBuilding

def train_and_transfer_sac():
    """
    Allenamento iniziale, trasferimento e fine-tuning continuo
    di un modello SAC tra ambienti diversi.
    """
    print("Fase 1: Addestramento iniziale su ambiente standard")

    # Ambiente di training iniziale
    train_env = create_custom_building_env(custom_model=NoisyLSTMDynamicsBuilding, noise_level=0.15)
    train_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(train_env))
    # Addestramento SAC iniziale
    train_env, sac_model = test_sac(train_env)
    # Crea una vera copia del modello originale usando save e load
    import os
    from tempfile import TemporaryDirectory
    
    # Crea una directory temporanea
    with TemporaryDirectory() as temp_dir:
        # Salva il modello originale
        original_model_path = os.path.join(temp_dir, "sac_original.zip")
        sac_model.save(original_model_path)
        
        # Carica una vera copia del modello
        from stable_baselines3 import SAC
        sac_original = SAC.load(original_model_path)

    print("Fase 2: Test e fine-tuning online su ambiente reale (diverso)")
    
    # Ambiente di test/applicazione (ambiente "reale" diverso)
    test_env = CityLearnEnv(**ENV_CONFIG_2)  
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
    
    print("Fase 3: Creazione e valutazione di un modello SAC con edifici adattivi")
    adaptive_b_env = create_custom_building_env(custom_model=AdaptiveLSTMDynamicsBuilding)
    adaptive_b_env= StableBaselines3Wrapper(NormalizedSpaceWrapper(adaptive_b_env))
    adaptive_b_env, sac_adaptive = test_sac(adaptive_b_env)
    
    #ambiente con molto rumore
    print("Fase 5: Test di un modello SAC con rumore")
    noisy_env = create_custom_building_env(custom_model=NoisyLSTMDynamicsBuilding, noise_level=0.45)
    noisy_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(noisy_env))
    noisy_env, sac_noisy = test_sac(noisy_env)

    #ambiente con media modificata
    print("Fase 6: Test di un modello SAC con media modificata")
    noisy_mean_env = create_custom_building_env(custom_model=NoisyLSTMDynamicsBuilding, noise_mean=1)
    noisy_mean_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(noisy_mean_env))
    noisy_mean_env, sac_noisy_mean = test_sac(noisy_mean_env)

    # Ambiente di valutazione standard
    eval_env = CityLearnEnv(**ENV_CONFIG_3)
    eval_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(eval_env))
    
    # Valuta i 4 modelli
    print("\nValutazione del modello originale senza adattamenti:")
    final_original_eval = evaluate_sac_performance_robust(eval_env, sac_original, "SAC-Original")
    
    print("\nValutazione del modello con fine-tuning tradizionale:")
    final_real_eval = evaluate_sac_performance_robust(eval_env, sac_model, "SAC-Transfer-Finetuned-Real")
    
    print("\nValutazione del modello con rumore:")
    final_noisy_eval = evaluate_sac_performance_robust(eval_env, sac_noisy, "SAC-Transfer-Finetuned-Noisy")

    print("\nValutazione del modello con media modificata:")
    final_noisy_mean_eval = evaluate_sac_performance_robust(eval_env, sac_noisy_mean, "SAC-Transfer-Finetuned-Noisy-Mean")

    print("\nValutazione del modello con edifici adattivi:")
    final_adaptive_eval = evaluate_sac_performance_robust(eval_env, sac_adaptive, "SAC-Adaptive-Building")
    
    # restituisco performance dei diversi ambienti
    results = {
        "real_env": final_real_eval,
        "original_env": final_original_eval,
        "noisy_env": final_noisy_eval,
        "noisy_mean_env": final_noisy_mean_eval,
        "adaptive_env": final_adaptive_eval,
    }

    print_model_params(sac_original, "SAC-Original")
    print_model_params(sac_model, "SAC-Fine-tuned") 
    print_model_params(sac_noisy, "SAC-Noisy")
    print_model_params(sac_noisy_mean, "SAC-Noisy-Mean")
    print_model_params(sac_adaptive, "SAC-Adaptive-Building")
    
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