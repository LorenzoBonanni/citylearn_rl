"""
costanti condivise tra vari moduli.
"""

# Configurazione dell'ambiente
DATASET_NAME = 'citylearn_challenge_2023_phase_3_1'
DATASET_NAME_2 = 'citylearn_challenge_2023_phase_3_2'
DATASET_NAME_3 = 'citylearn_challenge_2023_phase_3_3'
BUILDINGS = ['Building_1', 'Building_2', 'Building_3']
BUILDINGS_2 = ['Building_1', 'Building_2', 'Building_3','Building_4','Building_5']
SIMULATION_START_TIME_STEP = 0
SIMULATION_END_TIME_STEP = SIMULATION_START_TIME_STEP + 30 * 24 - 1 # Simula per 30 giorni

# Configurazione osservazioni e azioni
ACTIVE_OBSERVATIONS = ['hour', 'electricity_pricing', 'solar_generation', 'electricity_pricing', 
                      'net_electricity_consumption','electrical_storage_soc',
                      'indoor_dry_bulb_temperature', 'carbon_intensity']
ACTIVE_ACTIONS = ['electrical_storage', 'cooling_storage_soc', 'heating_storage_soc', 'cooling_or_heating_device']
CENTRAL_AGENT = True

# Altre costanti condivise
EPISODES = 10
SAVE_DIR = 'plots'
UPDATE_FREQ = 75

# Configurazione di ENV_CONFIG e ENV_CONFIG_2
ENV_CONFIG = {
    "schema": DATASET_NAME,
    "buildings": BUILDINGS_2,
    "simulation_start_time_step": SIMULATION_START_TIME_STEP,
    "simulation_end_time_step": SIMULATION_END_TIME_STEP,
    "active_observations": ACTIVE_OBSERVATIONS,
    "central_agent": CENTRAL_AGENT
}

ENV_CONFIG_2 = {
    "schema": DATASET_NAME_2,
    "buildings": BUILDINGS_2,
    "simulation_start_time_step": SIMULATION_START_TIME_STEP,
    "simulation_end_time_step": SIMULATION_END_TIME_STEP,
    "active_observations": ACTIVE_OBSERVATIONS,
    "central_agent": CENTRAL_AGENT
}

ENV_CONFIG_3 = {
    "schema": DATASET_NAME_3,
    "buildings": BUILDINGS_2,
    "simulation_start_time_step": SIMULATION_START_TIME_STEP,
    "simulation_end_time_step": SIMULATION_END_TIME_STEP,
    "active_observations": ACTIVE_OBSERVATIONS,
    "central_agent": CENTRAL_AGENT
}

# Configurazioni per gli agenti SAC
CUSTOM_AGENT_KWARGS = {
    "learning_rate": 0.0003,
    "tau": 0.005,
    "gamma": 0.99,
    "buffer_size": 10000,
    "learning_starts": 256,
    "batch_size": 256,
}