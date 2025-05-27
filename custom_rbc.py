from citylearn.agents.base import BaselineAgent
from citylearn.agents.rbc import RBC, HourRBC
from citylearn.building import Building
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from typing import Any, Mapping, List, Union
import pandas as pd
import os

MAX_TEMPERATURE = 25.0  # Temperatura massima consentita
MIN_TEMPERATURE = 21.0  # Temperatura minima consentita

class CustomRBC(HourRBC):
    
    def __init__(self, env: CityLearnEnv, action_map: Union[List[Mapping[str, Mapping[int, float]]], Mapping[str, Mapping[int, float]], Mapping[int, float]] = None, **kwargs: Any):
        super().__init__(env, action_map, **kwargs)
        # Inizializza il dataframe per memorizzare le temperature
        self.temperature_log = pd.DataFrame(columns=['time_step', 'building_id', 'temperature', 'set_point_cool', 'set_point_heat', 'action'])
    
    def predict_temperature(self, observations, deterministic = None):
        """
        Custom Predict to optimize temperature e registrare i dati
        """
        actions = []
        
        if self.action_map is None:
            actions = super().predict(observations, deterministic=deterministic)
        else:
            for i, (m, a, n, o) in enumerate(zip(self.action_map, self.action_names, self.observation_names, observations)):
                hour = o[n.index('hour')]
                actions_ = []
                building_id = self.env.buildings[i].name if i < len(self.env.buildings) else f"Building_{i}"
                
                # Registra la temperatura per questo building
                if 'indoor_dry_bulb_temperature' in n:
                    temp_idx = n.index('indoor_dry_bulb_temperature')
                    temperature = o[temp_idx]
                    
                    # Ottieni i setpoint dalla simulazione se disponibili
                    try:
                        set_point_cool = self.env.buildings[i].indoor_dry_bulb_temperature_cooling_set_point[-1]
                    except:
                        set_point_cool = None
                        
                    try:
                        set_point_heat = self.env.buildings[i].indoor_dry_bulb_temperature_heating_set_point[-1]
                    except:
                        set_point_heat = None
                    
                    # Registra le informazioni
                    action_value = None
                    
                for a_ in a:
                    if a_ == 'cooling_or_heating_device' and 'indoor_dry_bulb_temperature' in n:
                        temp_idx = n.index('indoor_dry_bulb_temperature')
                        temperature = o[temp_idx]
                        print(f"Building {building_id} - Time step {self.time_step} - Temperature: {temperature}")

                        if temperature > MAX_TEMPERATURE:
                            action_value = -0.5
                            m[a_][hour] = action_value
                        elif temperature < MIN_TEMPERATURE:
                            action_value = 0.5
                            m[a_][hour] = action_value
                            
                        # Salva nel log
                        new_row = {
                            'time_step': self.time_step,
                            'building_id': building_id,
                            'temperature': temperature,
                            'set_point_cool': set_point_cool,
                            'set_point_heat': set_point_heat,
                            'action': action_value
                        }
                        self.temperature_log = pd.concat([self.temperature_log, pd.DataFrame([new_row])], ignore_index=True)
                    
                    actions_.append(m[a_][hour])
                actions.append(actions_)
            self.actions = actions
            self.next_time_step()

        return actions
    
    def predict_temperature_bad(self, observations, deterministic = None):
        """
        Custom Predict per prestazioni di temperatura negative e registrazione dati
        """
        
        actions = []

        if self.action_map is None:
            actions = super().predict(observations, deterministic=deterministic)
        else:
            for i, (m, a, n, o) in enumerate(zip(self.action_map, self.action_names, self.observation_names, observations)):
                hour = o[n.index('hour')]
                actions_ = []
                building_id = self.env.buildings[i].name if i < len(self.env.buildings) else f"Building_{i}"
                
                for a_ in a:
                    if a_ == 'cooling_or_heating_device' and 'indoor_dry_bulb_temperature' in n:
                        temp_idx = n.index('indoor_dry_bulb_temperature')
                        temperature = o[temp_idx]
                        print(f"Building {building_id} - Time step {self.time_step} - Temperature: {temperature}")
                        
                        action_value = -1
                        m[a_][hour] = action_value

                        try:
                            set_point_cool = self.env.buildings[i].indoor_dry_bulb_temperature_cooling_set_point[-1]
                        except:
                            set_point_cool = None
                            
                        try:
                            set_point_heat = self.env.buildings[i].indoor_dry_bulb_temperature_heating_set_point[-1]
                        except:
                            set_point_heat = None
                        
                        new_row = {
                            'time_step': self.time_step,
                            'building_id': building_id,
                            'temperature': temperature,
                            'set_point_cool': set_point_cool,
                            'set_point_heat': set_point_heat,
                            'action': action_value
                        }
                        self.temperature_log = pd.concat([self.temperature_log, pd.DataFrame([new_row])], ignore_index=True)
                    
                    actions_.append(m[a_][hour])
                actions.append(actions_)
            self.actions = actions
            self.next_time_step()

        return actions
    
    def predict_multi_obj(self, observations, deterministic = None):
        actions = []
        if self.action_map is None:
            actions = super().predict(observations, deterministic=deterministic)
        else:
            for i, (m, a, n, o) in enumerate(zip(self.action_map, self.action_names, self.observation_names, observations)):
                hour = o[n.index('hour')]
                actions_ = []
                building_id = self.env.buildings[i].name if i < len(self.env.buildings) else f"Building_{i}"
                
                if 'net_electricity_consumption' in n:
                    elec_idx = n.index('net_electricity_consumption')
                    electricity_consumption = o[elec_idx]
                else:
                    electricity_consumption = 0

                if 'electricity_pricing' in n:
                    price_idx = n.index('electricity_pricing')
                    electricity_price = o[price_idx]
                else:
                    electricity_price = 0.15  # valore standard se non disponibile

                count=0
                num_o = len(n)/3

                for a_ in a:
                    count+=1
                    if a_ == 'electrical_storage' and 'solar_generation' in n:
                        gen_idx = n.index('solar_generation')#+int(count/3)*int(num_o)
                        generation = o[gen_idx]

                        if generation > 0:
                            if electricity_consumption > 1:
                                e_action_value = min(0.75, electricity_consumption * 0.5) 
                            else:   
                                e_action_value = 1
                        else:
                            if electricity_consumption > 1:
                                e_action_value = -1
                            else:   
                                e_action_value = -0.5
                            
                        #print(f"Building {building_id} - Time step {self.time_step} - Solar Generation: {generation:.2f} - Consumption: {electricity_consumption:.2f}- Action: {e_action_value}")

                        m[a_][hour] = e_action_value

                    elif a_ == 'cooling_or_heating_device' and 'indoor_dry_bulb_temperature' in n:
                        temp_idx = n.index('indoor_dry_bulb_temperature')
                        temperature = o[temp_idx]
             
                        try:
                            set_point_cool = self.env.buildings[i].indoor_dry_bulb_temperature_cooling_set_point[-1]-2
                        except:
                            set_point_cool = MIN_TEMPERATURE
                            
                        try:
                            set_point_heat = self.env.buildings[i].indoor_dry_bulb_temperature_heating_set_point[-1]+2
                        except:
                            set_point_heat = MAX_TEMPERATURE

                        price = 0.03
                        consumption = 1
                        ultra_soft_action = 0.2
                        soft_action = 0.4
                        normal_action = 0.6
                        strong_action = 1
                        
                        if temperature > set_point_heat:
                            if electricity_price > price:
                                action_value = -soft_action
                                if electricity_consumption > consumption:
                                    action_value = -ultra_soft_action
                            else: #costo elettrico basso
                                action_value = -normal_action
                            if temperature > set_point_heat+2:
                                action_value = -strong_action
                        elif temperature < set_point_cool:
                            if electricity_price > price:
                                action_value = soft_action
                                if electricity_consumption > consumption:
                                    action_value = ultra_soft_action
                            else:
                                action_value = normal_action
                            if temperature < set_point_cool-2:
                                action_value = strong_action
                        else:
                            action_value = 0.0

                        m[a_][hour] = action_value

                        #print(f"Building {building_id} - Time {hour} - Indoor_dry_bulb: {set_point_cool+2:.2f}°C -Temp: {temperature:.1f}°C - Price: {electricity_price:.2f} - Action:{action_value}")
                        
                        new_row = {
                            'time_step': self.time_step,
                            'building_id': building_id,
                            'temperature': temperature,
                            'set_point_cool': set_point_cool,
                            'set_point_heat': set_point_heat,
                            'action': action_value
                        }
                        self.temperature_log = pd.concat([self.temperature_log, pd.DataFrame([new_row])], ignore_index=True)
                    
                    actions_.append(m[a_][hour])
                actions.append(actions_)
            self.actions = actions
            self.next_time_step()
        
        return actions

    def _save_temperature_log(self, filename: str = None):
        """Salva il log delle temperature in un file CSV"""
        # Crea la directory se non esiste
        os.makedirs('logs', exist_ok=True)

        if filename is None:
            filename = f'logs/temperature_log_{self.__class__.__name__}.csv'
        else:
            filename = os.path.join('logs', filename)
        self.temperature_log.to_csv(filename, index=False)
        print(f"Saved temperature log to {filename}")
        
    def finalize(self):
        """Da chiamare alla fine della simulazione per salvare i dati rimanenti"""
        self._save_temperature_log()