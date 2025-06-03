"""
Test comparativo tra training offline tradizionale e fine-tuning online.
Struttura più pulita e modulare rispetto a test_online_tuning.py
"""

from functools import partial
from custom_function import *
from constants import RANDOM_SEED  # Import RANDOM_SEED
import os
import numpy as np
from gymnasium.wrappers import TransformObservation
from adaptive_building import AdaptiveLSTMDynamicsBuilding
from tempfile import TemporaryDirectory
from stable_baselines3 import SAC
import time
from datetime import datetime

def train_offline_models():
    """
    Allena tutti i modelli offline con diversi tipi di rumore.
    
    Returns:
    dict: Dizionario contenente modelli allenati e dati di training
    """
    print("="*60)
    print("FASE 1: TRAINING OFFLINE DEI MODELLI")
    print("="*60)
    
    models = {}
    training_data = {}
    
    print("\n1. Training modello NORMALE (baseline)")
    normal_env = CityLearnEnv(**ENV_CONFIG)
    normal_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(normal_env))

    _, sac_normal, training_rewards_normal, timesteps_normal = train_sac(
        normal_env, 
        track_rewards=True,
        eval_freq=50  
    )
    
    models['normal'] = sac_normal
    training_data['normal'] = {
        'rewards': training_rewards_normal, 
        'timesteps': timesteps_normal,
        'env': normal_env
    }
    
    print("\n2. Training modello con POCO RUMORE")
    noisy_env = CityLearnEnv(**ENV_CONFIG)
    noisy_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(noisy_env))
    observation, _ = noisy_env.reset()
    noisy_env = TransformObservation(noisy_env, partial(
            add_noise_to_observations, 
            noise_type='gaussian', 
            noise_level=0.05, 
            noise_mean=0.0
        ))
    
    _, sac_noisy, training_rewards_noisy, timesteps_noisy = train_sac(
        noisy_env, 
        track_rewards=True, 
        eval_freq=50  
    )
    
    models['noisy'] = sac_noisy
    training_data['noisy'] = {
        'rewards': training_rewards_noisy, 
        'timesteps': timesteps_noisy,
        'env': noisy_env
    }
    
    print("\n3. Training modello con MOLTO RUMORE")
    more_noise_env = CityLearnEnv(**ENV_CONFIG)
    more_noise_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(more_noise_env))
    more_noise_env = TransformObservation(more_noise_env, partial(
            add_noise_to_observations, 
            noise_type='gaussian', 
            noise_level=0.15, 
            noise_mean=0.0
        ))
    
    _, sac_more_noise, training_rewards_more_noise, timesteps_more_noise = train_sac(
        more_noise_env,
        track_rewards=True,
        eval_freq=50
    )
    
    models['more_noisy'] = sac_more_noise
    training_data['more_noisy'] = {
        'rewards': training_rewards_more_noise, 
        'timesteps': timesteps_more_noise,
        'env': more_noise_env
    }
    
    print("\n4. Training modello con MEDIA MODIFICATA")
    noisy_mean_env = CityLearnEnv(**ENV_CONFIG)
    noisy_mean_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(noisy_mean_env))
    noisy_mean_env = TransformObservation(noisy_mean_env, partial(
            add_noise_to_observations, 
            noise_type='gaussian', 
            noise_level=0.05, 
            noise_mean=0.2
        ))
    
    _, sac_noisy_mean, training_rewards_noisy_mean, timesteps_noisy_mean = train_sac(
        noisy_mean_env,
        track_rewards=True,
        eval_freq=50
    )
    
    models['noisy_mean'] = sac_noisy_mean
    training_data['noisy_mean'] = {
        'rewards': training_rewards_noisy_mean, 
        'timesteps': timesteps_noisy_mean,
        'env': noisy_mean_env
    }
    
    print("\n✅ Training offline completato!")
    return models, training_data


def apply_online_finetuning(models, target_env):
    """
    Applica fine-tuning online ai modelli pre-allenati.
    
    Parametri:
    models: dict - Modelli pre-allenati
    target_env: CityLearnEnv - Ambiente target per il fine-tuning
    
    Returns:
    dict: Modelli con fine-tuning applicato
    """
    print("="*60)
    print("FASE 2: FINE-TUNING ONLINE")
    print("="*60)
    
    finetuned_models = {}
    online_training_data = {}
    
    for model_name, model in models.items():
        print(f"\n🔄 Fine-tuning del modello: {model_name.upper()}")
        
        # Crea una copia del modello per il fine-tuning
        with TemporaryDirectory() as temp_dir:
            # Salva il modello originale
            model_path = os.path.join(temp_dir, f"model_{model_name}.zip")
            model.save(model_path)
            
            # Carica una copia per il fine-tuning
            finetuned_model = SAC.load(model_path)
            finetuned_model.set_env(target_env)
        
        # Applica online learning
        online_results = simple_online_learning(
            env=target_env,
            model=finetuned_model,
            update_freq=4, 
            batch_size=32,    
            gradient_steps=5, 
            verbose=1
        )
        
        finetuned_models[f"{model_name}_finetuned"] = finetuned_model
        online_training_data[f"{model_name}_finetuned"] = {
            'step_rewards': online_results['step_rewards'],
            'total_reward': online_results['total_reward'],
            'final_observations': online_results['final_observations']
        }
        
        target_env.reset()  # Reset per il prossimo modello
    
    print("\n✅ Fine-tuning online completato!")
    return finetuned_models, online_training_data


def evaluate_all_models(original_models, finetuned_models, test_env):
    """
    Valuta tutti i modelli (originali e con fine-tuning) sull'ambiente di test.
    
    Parametri:
    original_models: dict - Modelli originali
    finetuned_models: dict - Modelli con fine-tuning
    test_env: CityLearnEnv - Ambiente di test
    
    Returns:
    dict: Risultati di valutazione per tutti i modelli
    """
    print("="*60)
    print("FASE 3: VALUTAZIONE COMPARATIVA")
    print("="*60)
    
    evaluation_results = {}
    
    # Valuta modelli originali
    print("\n📊 Valutazione modelli ORIGINALI:")
    for model_name, model in original_models.items():
        print(f"\n🔍 Valutando {model_name.upper()}...")
        result = evaluate_sac_performance(test_env, model, f"Original-{model_name}")
        evaluation_results[f"Original-{model_name}"] = result
        test_env.reset()
    
    # Valuta modelli con fine-tuning
    print("\n📊 Valutazione modelli FINE-TUNED:")
    for model_name, model in finetuned_models.items():
        print(f"\n🔍 Valutando {model_name.upper()}...")
        result = evaluate_sac_performance(test_env, model, f"Finetuned-{model_name}")
        evaluation_results[f"Finetuned-{model_name}"] = result
        test_env.reset()
    
    return evaluation_results


def plot_models_separately(models_data, model_type, save_dir):
    """
    Funzione generica per plottare un gruppo di modelli (originali o fine-tuned).
    
    Parametri:
    models_data: dict - Dati dei modelli da plottare
    model_type: str - Tipo di modelli ("Original" o "Finetuned")
    save_dir: str - Directory di salvataggio
    """
    if not models_data:
        print(f"⚠️ Nessun dato per modelli {model_type}")
        return
    
    print(f"📈 Generando grafico per modelli {model_type}...")
    
    # Genera il grafico usando la funzione esistente
    cp.plot_post_training_rewards(
        models_data, 
        save_dir=save_dir,
        title_suffix=f" - {model_type} Models"
    )
    
    # Salva con nome specifico
    filename = f"models_{model_type.lower()}_comparison.png"
    print(f"💾 Grafico salvato: {filename}")


def save_results(evaluation_results, training_data, online_training_data, save_dir):
    """
    Salva tutti i risultati e genera grafici separati.
    
    Parametri:
    evaluation_results: dict - Risultati di valutazione
    training_data: dict - Dati di training offline
    online_training_data: dict - Dati di training online
    save_dir: str - Directory di salvataggio
    """
    print("="*60)
    print("FASE 4: SALVATAGGIO RISULTATI")
    print("="*60)
    
    # Separa i risultati per tipo
    offline_results = {k: v for k, v in evaluation_results.items() if 'Original' in k}
    online_results = {k: v for k, v in evaluation_results.items() if 'Finetuned' in k}
    
    print(f"\n💾 Salvando risultati in: {save_dir}")
    
    # 1. Grafici separati per modelli originali e fine-tuned
    plot_models_separately(offline_results, "Original", save_dir)
    plot_models_separately(online_results, "Finetuned", save_dir)
    
    # 2. Grafico combinato (opzionale)
    #print("📈 Generando grafico combinato...")
    #cp.plot_post_training_rewards(evaluation_results, save_dir=save_dir)
    
    # 3. Grafici individuali per ogni modello fine-tuned
    # print("📈 Generando grafici fine-tuning individuali...")
    # for model_name, data in online_training_data.items():
    #     online_results_single = {
    #         'step_rewards': data['step_rewards'],
    #         'total_reward': data['total_reward']
    #     }
    #     cp.plot_online_fine_tuning_simple(
    #         online_results_single,
    #         save_dir=save_dir,
    #         title=f"Fine-tuning Online: {model_name.replace('_', ' ').title()}"
    #     )
    
    # 4. Salva riepilogo testuale con calcolo miglioramento dettagliato
    print("📝 Generando riepilogo testuale...")
    summary_path = os.path.join(save_dir, "experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("ESPERIMENTO: CONFRONTO OFFLINE vs ONLINE LEARNING\n")
        f.write("="*80 + "\n")
        f.write(f"Data esperimento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"RANDOM_SEED utilizzato: {RANDOM_SEED}\n\n")  # Aggiunto RANDOM_SEED
        
        f.write("RISULTATI MODELLI ORIGINALI (OFFLINE):\n")
        f.write("-" * 50 + "\n")
        for model_name, results in offline_results.items():
            f.write(f"• {model_name}: {results['total_reward']:.2f}\n")
        
        f.write("\nRISULTATI MODELLI FINE-TUNED (ONLINE):\n")
        f.write("-" * 50 + "\n")
        for model_name, results in online_results.items():
            f.write(f"• {model_name}: {results['total_reward']:.2f}\n")
        
        f.write("\nCONFRONTI MIGLIORAMENTO:\n")
        f.write("-" * 50 + "\n")
                # Confronta ogni modello originale con la sua versione fine-tuned
        for orig_key in offline_results.keys():
            base_name = orig_key.replace('Original-', '')
            finetuned_key = f"Finetuned-{base_name}_finetuned"
            
            if finetuned_key in online_results:
                orig_reward = offline_results[orig_key]['total_reward']
                finetuned_reward = online_results[finetuned_key]['total_reward']
                                
                if abs(orig_reward) > 0:
                    improvement = ((finetuned_reward - orig_reward) / abs(orig_reward)) * 100
                    f.write(f"• {base_name}: {improvement:+.2f}% "
                           f"({orig_reward:.2f} → {finetuned_reward:.2f})\n")
                    
    print(f"✅ Riepilogo salvato: {summary_path}")


def main():
    """Funzione principale che coordina l'intero esperimento."""
    
    print("🚀 AVVIO ESPERIMENTO: OFFLINE vs ONLINE LEARNING")
    print("="*80)
    
    # Setup directory di salvataggio
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(SAVE_DIR, f"offline_vs_online_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"📁 Directory risultati: {run_dir}")
    
    start_time = time.time()
    
    try:
        # 1. Training offline dei modelli
        original_models, training_data = train_offline_models()
        
        # 2. Setup ambiente target per fine-tuning (ambiente pulito)
        target_env = CityLearnEnv(**ENV_CONFIG)
        target_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(target_env))
        
        # 3. Fine-tuning online
        finetuned_models, online_training_data = apply_online_finetuning(
            original_models, target_env
        )
        
        # 4. Valutazione comparativa
        evaluation_results = evaluate_all_models(
            original_models, finetuned_models, target_env
        )
        
        # 5. Analisi e stampa risultati
        print("="*60)
        print("ANALISI RISULTATI FINALI")
        print("="*60)
        
        # Stampa confronti
        offline_results = {k: v for k, v in evaluation_results.items() if 'Original' in k}
        online_results = {k: v for k, v in evaluation_results.items() if 'Finetuned' in k}
        
        print("\n📊 CONFRONTO PERFORMANCE:")
        print("-" * 40)
        for orig_key in offline_results.keys():
            base_name = orig_key.replace('Original-', '')
            finetuned_key = f"Finetuned-{base_name}_finetuned"
            
            if finetuned_key in online_results:
                orig_reward = offline_results[orig_key]['total_reward']
                finetuned_reward = online_results[finetuned_key]['total_reward']
                
                print(f"\n{base_name.upper()}:")
                print(f"  Offline:  {orig_reward:.2f}")
                print(f"  Online:   {finetuned_reward:.2f}")
                
                if abs(orig_reward) > 0:
                    improvement = ((finetuned_reward - orig_reward) / abs(orig_reward)) * 100
                    direction = "↗️" if improvement > 0 else "↘️"
                    print(f"  Cambio:   {improvement:+.2f}% {direction}")
        
        # 6. Salvataggio risultati e grafici
        save_results(evaluation_results, training_data, online_training_data, run_dir)
        
        # 7. Debug parametri modelli
        print("\n🔍 DEBUG PARAMETRI MODELLI:")
        for name, model in original_models.items():
            print_model_params(model, f"Original-{name}")
        
        for name, model in finetuned_models.items():
            print_model_params(model, f"Finetuned-{name}")
            
    except Exception as e:
        print(f"\n❌ ERRORE durante l'esperimento: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n⏱️ Esperimento completato in {duration/60:.2f} minuti ({duration:.1f} secondi)")
        print(f"📁 Risultati salvati in: {run_dir}")


if __name__ == "__main__":
    main()