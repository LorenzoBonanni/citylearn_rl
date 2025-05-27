import torch
import numpy as np
from typing import Any, List, Mapping, Union
from citylearn.building import LSTMDynamicsBuilding, Building
from citylearn.dynamics import LSTMDynamics, Dynamics
from citylearn.citylearn import CityLearnEnv

class CustomLSTMDynamics(LSTMDynamics):
    """
    Classe personalizzata che estende LSTMDynamics per consentire la modifica 
    del comportamento del modello dinamico LSTM.
    
    Parameters
    ----------
    base_dynamics: LSTMDynamics
        Il modello LSTM di base che viene esteso/modificato
    temperature_offset: float, default: 0.0
        Offset di temperatura da applicare alla predizione (in gradi C)
    scale_factor: float, default: 1.0
        Fattore di scala da applicare alla predizione
    custom_forward: callable, optional
        Funzione personalizzata da utilizzare invece del metodo forward standard
    """
    
    def __init__(
        self, 
        base_dynamics: LSTMDynamics,
        temperature_offset: float = 0.0,
        scale_factor: float = 1.0,
        custom_forward = None
    ):
        super().__init__(
            filepath=base_dynamics.filepath,
            input_observation_names=base_dynamics.input_observation_names,
            input_normalization_minimum=base_dynamics.input_normalization_minimum,
            input_normalization_maximum=base_dynamics.input_normalization_maximum,
            hidden_size=base_dynamics.hidden_size,
            num_layers=base_dynamics.num_layers,
            lookback=base_dynamics.lookback,
            input_size=base_dynamics.input_size
        )
        
        self.temperature_offset = temperature_offset
        self.scale_factor = scale_factor
        self.custom_forward = custom_forward
        self.base_dynamics = base_dynamics
        
        with torch.no_grad():
            self.l_lstm.weight_ih_l0.copy_(base_dynamics.l_lstm.weight_ih_l0)
            self.l_lstm.weight_hh_l0.copy_(base_dynamics.l_lstm.weight_hh_l0)
            self.l_lstm.bias_ih_l0.copy_(base_dynamics.l_lstm.bias_ih_l0)
            self.l_lstm.bias_hh_l0.copy_(base_dynamics.l_lstm.bias_hh_l0)
            self.l_lstm.weight_ih_l1.copy_(base_dynamics.l_lstm.weight_ih_l1)
            self.l_lstm.weight_hh_l1.copy_(base_dynamics.l_lstm.weight_hh_l1)
            self.l_lstm.bias_ih_l1.copy_(base_dynamics.l_lstm.bias_ih_l1)
            self.l_lstm.bias_hh_l1.copy_(base_dynamics.l_lstm.bias_hh_l1)
            self.l_linear.weight.copy_(base_dynamics.l_linear.weight)
            self.l_linear.bias.copy_(base_dynamics.l_linear.bias)
    
    def forward(self, x, h):
        """
        Override del metodo forward per modificare la predizione del modello.
        """
        if self.custom_forward is not None:
            return self.custom_forward(self, x, h)
        
        output, hidden_state = super().forward(x, h)
        modified_output = output * self.scale_factor
        
        return modified_output, hidden_state
    
    def reset(self):
        """Reset del modello personalizzato."""
        self._hidden_state = self.init_hidden(1)
        self._model_input = [[None]*(self.lookback + 1) for _ in self.input_observation_names]
    
    def set_parameters(self, temperature_offset=None, scale_factor=None, custom_forward=None):
        """
        Aggiorna i parametri del modello personalizzato.
        """
        if temperature_offset is not None:
            self.temperature_offset = temperature_offset
        if scale_factor is not None:
            self.scale_factor = scale_factor
        if custom_forward is not None:
            self.custom_forward = custom_forward


class CustomLSTMDynamicsBuilding(LSTMDynamicsBuilding):
    """
    Building personalizzato che utilizza un modello LSTM modificabile per
    predire la dinamica della temperatura interna.
    
    Parameters
    ----------
    *args: Any
        Argomenti posizionali per LSTMDynamicsBuilding
    dynamics: LSTMDynamics
        Modello di dinamica LSTM
    temperature_offset: float, default: 0.0
        Offset di temperatura da applicare alla predizione
    dynamics_params: dict, optional
        Parametri aggiuntivi per il modello di dinamica personalizzato
    **kwargs: Any
        Altri argomenti keyword per LSTMDynamicsBuilding
    """
    
    def __init__(
        self, 
        *args: Any, 
        dynamics: LSTMDynamics,
        temperature_offset: float = 0.0,
        dynamics_params: dict = None,
        **kwargs: Any
    ):
        if not isinstance(dynamics, CustomLSTMDynamics):
            self.custom_dynamics = CustomLSTMDynamics(
                base_dynamics=dynamics,
                temperature_offset=temperature_offset,
                **(dynamics_params or {})
            )
        else:
            self.custom_dynamics = dynamics
            
        super().__init__(*args, dynamics=self.custom_dynamics, **kwargs)
    
    def update_indoor_dry_bulb_temperature(self):
        """
        Override del metodo per predire e aggiornare la temperatura interna.
        """
        super().update_indoor_dry_bulb_temperature()
        
        if self.custom_dynamics.temperature_offset != 0:
            self.energy_simulation.indoor_dry_bulb_temperature[self.time_step] += self.custom_dynamics.temperature_offset
    
    def modify_dynamics(self, temperature_offset=None, scale_factor=None, custom_forward=None):
        """
        Modifica il comportamento del modello dinamico.
        """
        self.custom_dynamics.set_parameters(
            temperature_offset=temperature_offset,
            scale_factor=scale_factor,
            custom_forward=custom_forward
        )
    
    def reset(self):
        """Reset del building e del modello dinamico."""
        super().reset()