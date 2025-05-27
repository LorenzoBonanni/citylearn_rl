from citylearn.reward_function import RewardFunction
from typing import Any, Mapping, List, Union
import numpy as np
import pandas as pd

class CustomReward(RewardFunction):
    def __init__(self, env_metadata: dict[str, Any] = None, **kwargs):
        """Initialize CustomReward.

        Parameters
        ----------
        env_metadata: dict[str, Any]:
            General static information about the environment.
        """

        super().__init__(env_metadata)

    def calculate(
        self, observations: list[dict[str, int | float]]
    ) -> list[float]:
        """Returns reward for most recent action.

        The reward is designed to minimize electricity cost.
        It is calculated for each building, i and summed to provide the agent
        with a reward that is representative of all n buildings.
        """

        reward_list = []

        for o, m in zip(observations, self.env_metadata['buildings']):
            # 1. Costo elettricità (priorità massima)
            cost = o['net_electricity_consumption'] * o['electricity_pricing']
            
            # 2. Uso della batteria in relazione alla generazione solare
            battery_soc = o['electrical_storage_soc']
            solar_generation = o.get('solar_generation', 0.0)
            
            # 3. Comfort termico e discomfort da sbalzi di temperatura
            indoor_temp = o.get('indoor_dry_bulb_temperature', 0.0)
            
            # Ottieni le temperature di setpoint per calcolare il comfort
            cooling_setpoint = o.get('indoor_dry_bulb_temperature_cooling_set_point', 24.0)
            heating_setpoint = o.get('indoor_dry_bulb_temperature_heating_set_point', 20.0)
            comfort_band = 2.0
            
            # Calcola i limiti di comfort
            upper_limit = cooling_setpoint + comfort_band
            lower_limit = heating_setpoint - comfort_band
            
            # Calcola il discomfort termico
            temp_discomfort = 0.0
            if indoor_temp > upper_limit:
                temp_discomfort = -(indoor_temp - upper_limit) ** 2
            elif indoor_temp < lower_limit:
                temp_discomfort = -(lower_limit - indoor_temp) ** 2
            
            # 4. Emissioni
            emissions = o['net_electricity_consumption'] * o['carbon_intensity']
            
            # Pesi relativi in base alle priorità
            w_cost = 1.0
            w_battery = 1.0
            w_comfort = 0.5
            w_emissions = 0.3
            w_solar = 0.8
            
            # Calcola i diversi componenti del reward
            cost_component = -abs(cost) * w_cost
            
            if solar_generation > 0:
                # Durante il giorno, premiamo la carica della batteria
                battery_solar_component = (1.0 - battery_soc) * solar_generation * w_solar
                # Penalizziamo l'esportazione di energia se la batteria non è carica
                battery_component = -(1.0 + np.sign(cost) * battery_soc) * abs(cost) * w_battery
            else:
                # Durante la notte, premiamo l'uso della batteria per ridurre il consumo
                battery_solar_component = -battery_soc * abs(cost) * w_solar
                battery_component = -(1.0 + np.sign(cost) * battery_soc) * abs(cost) * w_battery
            
            comfort_component = temp_discomfort * w_comfort
            emissions_component = -abs(emissions) * w_emissions
            
            # Combina tutti i componenti del reward
            reward = cost_component + battery_component + battery_solar_component + comfort_component + emissions_component
            reward_list.append(reward)

        # Se stiamo usando un agente centrale, sommiamo tutti i reward
        reward = [sum(reward_list)]

        return reward