from custom_function import *

if __name__ == "__main__":

    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    warnings.filterwarnings('ignore', category=DeprecationWarning)
      # L'approccio alternativo non richiede patching dei metodi Building
    # Commentiamo questa linea per usare l'approccio robusto
    # patch_building_methods()

    #print("Inizio simulazione Baseline")
    base_env = CityLearnEnv(**ENV_CONFIG)

    # baseline_model = BaselineAgent(base_env)
    # observations, _ = base_env.reset()
    # while not base_env.terminated:
    #     actions = baseline_model.predict(observations)
    #     observations, _, _, _, _ = base_env.step(actions)

    # print("Inizio simulazione RBC")
    # env3 = CityLearnEnv(**ENV_CONFIG)
    # kpis3, env3 = test_rbc(env3,3)
    print("Inizio simulazione SAC standard")
    env_2 = CityLearnEnv(**ENV_CONFIG)
    sac_env_2 = StableBaselines3Wrapper(NormalizedObservationWrapper(env_2))
    sac_env_2, sac_model_2 = test_sac(sac_env_2)

    print("Inizio simulazione SAC + RBC standard")
    env_2_rbc = CityLearnEnv(**ENV_CONFIG)
    sac_env_2_rbc = StableBaselines3Wrapper(NormalizedObservationWrapper(env_2_rbc))
    sac_env_2_rbc, sac_model_2_rbc = test_sac(sac_env_2_rbc, n=2)

    # print("Inizio simulazione SAC con edifici più caldi (+2°C)")
    # env_3 = create_custom_building_env(temperature_offset=2.0, scale_factor=1.0)
    # sac_env_3 = StableBaselines3Wrapper(NormalizedObservationWrapper(env_3))
    # sac_env_3, sac_model_3 = test_sac(sac_env_3)
    
    # print("Inizio simulazione SAC + RBC con edifici più reattivi (1.5x)")
    # env_4 = create_custom_building_env(temperature_offset=0.0, scale_factor=1.5)
    # sac_env_4 = StableBaselines3Wrapper(NormalizedObservationWrapper(env_4))
    # sac_env_4, sac_model_4 = test_sac(sac_env_4, n=2)

    print("Inizio simulazione SAC con edifici Noisy")
    env_noisy = create_custom_building_env(custom_model=NoisyLSTMDynamicsBuilding)
    sac_env_noisy = StableBaselines3Wrapper(NormalizedObservationWrapper(env_noisy))
    sac_env_noisy, sac_model_noisy = test_sac(sac_env_noisy)

    print("Inizio simulazione SAC + RBC con edifici Noisy")
    env_noisy_rbc = create_custom_building_env(custom_model=NoisyLSTMDynamicsBuilding)
    sac_env_noisy_rbc = StableBaselines3Wrapper(NormalizedObservationWrapper(env_noisy_rbc))
    sac_env_noisy_rbc, sac_model_noisy_rbc = test_sac(sac_env_noisy_rbc, n=2)

    envs = {
        'Baseline': base_env,
        #'Multi_Obj': env3,
        'SAC-Standard': sac_env_2.unwrapped,
        'SAC-Noisy': sac_env_noisy.unwrapped,
        # 'SAC-Custom-Hot': sac_env_3.unwrapped,
        # 'SAC-Custom-Reactive': sac_env_4.unwrapped,
        'SAC-RBC-Standard': sac_env_2_rbc.unwrapped,
        'SAC-RBC-Noisy': sac_env_noisy_rbc.unwrapped
    }

    #cp.plot_simulation_summary(envs, standard_baseline_env=base_env, save_dir=SAVE_DIR)

    reward_envs = {
        'SAC-Standard': pd.DataFrame(sac_env_2.unwrapped.episode_rewards)['sum'].tolist(),
        'SAC-RBC-Standard': pd.DataFrame(sac_env_2_rbc.unwrapped.episode_rewards)['sum'].tolist(),
        # 'SAC-Custom-Hot': pd.DataFrame(sac_env_3.unwrapped.episode_rewards)['sum'].tolist(),
        # 'SAC-Custom-Reactive': pd.DataFrame(sac_env_4.unwrapped.episode_rewards)['sum'].tolist(),
        'SAC-Noisy': pd.DataFrame(sac_env_noisy.unwrapped.episode_rewards)['sum'].tolist(),
        'SAC-RBC-Noisy': pd.DataFrame(sac_env_noisy_rbc.unwrapped.episode_rewards)['sum'].tolist(),
    }

    cp.plot_reward_summary(reward_envs, save_dir=SAVE_DIR)

    print("Valutazione post-training delle prestazioni dei modelli...")
    
    # nuovo ambiente per la valutazione
    eval_env_standard = CityLearnEnv(**ENV_CONFIG)
    eval_env_standard = StableBaselines3Wrapper(NormalizedObservationWrapper(eval_env_standard))

    evaluation_results = {}
    
    try:
        evaluation_results['SAC-Standard'] = evaluate_sac_performance_robust(eval_env_standard, sac_model_2, "SAC-Standard")
        evaluation_results['SAC-RBC-Standard'] = evaluate_sac_performance_robust(eval_env_standard, sac_model_2_rbc, "SAC-RBC-Standard")
        evaluation_results['SAC-Noisy'] = evaluate_sac_performance_robust(eval_env_standard, sac_model_noisy, "SAC-Noisy")
        evaluation_results['SAC-RBC-Noisy'] = evaluate_sac_performance_robust(eval_env_standard, sac_model_noisy_rbc, "SAC-RBC-Noisy")
        
        cp.plot_post_training_rewards(
            evaluation_results,
            save_dir=SAVE_DIR,
        )

    except Exception as e:
        print(f"Errore durante la valutazione post-training: {e}")
