"""
costanti condivise tra vari moduli.
"""

# Configurazione dell'ambiente
from citylearn.reward_function import SolarPenaltyAndComfortReward


DATASET_NAME = 'citylearn_challenge_2023_phase_3_1'
DATASET_NAME_2 = 'citylearn_challenge_2023_phase_3_2'
DATASET_NAME_3 = 'citylearn_challenge_2023_phase_3_3'
BUILDINGS = ['Building_1', 'Building_2', 'Building_3']
BUILDINGS_2 = ['Building_1', 'Building_2', 'Building_3','Building_4','Building_5']
SIMULATION_START_TIME_STEP = 0
SIMULATION_END_TIME_STEP = 1709 # Simula per 1709 ore (circa 71 giorni)

# Configurazione osservazioni e azioni
ACTIVE_OBSERVATIONS = ['hour', 'electricity_pricing', 'solar_generation', 'electricity_pricing', 
                      'net_electricity_consumption','electrical_storage_soc',
                      'indoor_dry_bulb_temperature', 'carbon_intensity']
ACTIVE_ACTIONS = ['electrical_storage', 'cooling_storage_soc', 'heating_storage_soc', 'cooling_or_heating_device']
CENTRAL_AGENT = True

# Altre costanti condivise
EPISODES = 2
SAVE_DIR = 'plots'
UPDATE_FREQ = 75
BATCH_SIZE = 256
LEARNING_STARTS = 150
RANDOM_SEED = 42

# Configurazione di ENV_CONFIG e ENV_CONFIG_2
ENV_CONFIG = {
    "schema": 'citylearn_challenge_2023_phase_1',
    "central_agent": CENTRAL_AGENT,
    'reward_function': SolarPenaltyAndComfortReward,
}

ENV_CONFIG_2 = {
    "schema": 'citylearn_challenge_2023_phase_2_local_evaluation',
    "central_agent": CENTRAL_AGENT,
    'reward_function': SolarPenaltyAndComfortReward
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
	"batch_size": BATCH_SIZE,
    "learning_starts": LEARNING_STARTS,
}