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
        # Inizializza usando i parametri del modello di base
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
        
        # Parametri di modifica del comportamento
        self.temperature_offset = temperature_offset
        self.scale_factor = scale_factor
        self.custom_forward = custom_forward
        
        # Conserva riferimento al modello originale
        self.base_dynamics = base_dynamics
        
        # Copiamo immediatamente i pesi dal modello base
        # Questo è essenziale per evitare problemi durante il reset
        with torch.no_grad():
            self.l_lstm.load_state_dict(base_dynamics.l_lstm.state_dict())
            self.l_linear.load_state_dict(base_dynamics.l_linear.state_dict())
    
    def forward(self, x, h):
        """
        Override del metodo forward per modificare la predizione del modello.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor al modello LSTM
        h: tuple
            Stato nascosto del modello LSTM
            
        Returns
        -------
        tuple
            (output, hidden_state) - output modificato e nuovo stato nascosto
        """
        if self.custom_forward is not None:
            # Usa la funzione forward personalizzata se fornita
            return self.custom_forward(self, x, h)
        
        # Altrimenti usa il forward di base con le modifiche specificate
        output, hidden_state = super().forward(x, h)
        
        # Applica le modifiche alla predizione normalizzata (tra 0 e 1)
        # Nota: fare attenzione alla normalizzazione
        modified_output = output * self.scale_factor
        
        # L'offset di temperatura verrà applicato dopo la denormalizzazione
        # nella funzione update_indoor_dry_bulb_temperature del building
        
        return modified_output, hidden_state
    
    def reset(self):
        """Reset del modello personalizzato."""
        # Non carichiamo il modello dal file, usiamo solo le strutture dati necessarie
        self._hidden_state = self.init_hidden(1)
        self._model_input = [[None]*(self.lookback + 1) for _ in self.input_observation_names]
    
    def set_parameters(self, temperature_offset=None, scale_factor=None, custom_forward=None):
        """
        Aggiorna i parametri del modello personalizzato.
        
        Parameters
        ----------
        temperature_offset: float, optional
            Nuovo offset di temperatura
        scale_factor: float, optional
            Nuovo fattore di scala
        custom_forward: callable, optional
            Nuova funzione forward personalizzata
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
        # Assicuriamoci che electric_vehicle_chargers sia una lista vuota se non presente
        if 'electric_vehicle_chargers' not in kwargs or kwargs['electric_vehicle_chargers'] is None:
            kwargs['electric_vehicle_chargers'] = []
            
        # Inizializza il modello di dinamica personalizzato
        # Ma prima eseguiamo un reset sul modello base per assicurarci che sia caricato
        dynamics.reset()
        
        # Crea il modello di dinamica personalizzato
        self.custom_dynamics = CustomLSTMDynamics(
            base_dynamics=dynamics,
            temperature_offset=temperature_offset,
            **(dynamics_params or {})
        )
            
        # Inizializza il building con il modello personalizzato
        super().__init__(*args, dynamics=self.custom_dynamics, **kwargs)
    
    def update_indoor_dry_bulb_temperature(self):
        """
        Override del metodo per predire e aggiornare la temperatura interna.
        Estende il comportamento standard con la possibilità di applicare un offset.
        """
        # Chiama il metodo standard di LSTMDynamicsBuilding
        super().update_indoor_dry_bulb_temperature()
        
        # Applica l'offset alla temperatura dopo la denormalizzazione
        if self.custom_dynamics.temperature_offset != 0:
            self.energy_simulation.indoor_dry_bulb_temperature[self.time_step] += self.custom_dynamics.temperature_offset
    
    def modify_dynamics(self, temperature_offset=None, scale_factor=None, custom_forward=None):
        """
        Modifica il comportamento del modello dinamico.
        
        Parameters
        ----------
        temperature_offset: float, optional
            Nuovo offset di temperatura
        scale_factor: float, optional
            Nuovo fattore di scala per la predizione
        custom_forward: callable, optional
            Funzione forward personalizzata
        """
        self.custom_dynamics.set_parameters(
            temperature_offset=temperature_offset,
            scale_factor=scale_factor,
            custom_forward=custom_forward
        )
    
    def reset(self):
        """Reset del building e del modello dinamico."""
        super().reset()
        # Eventuali operazioni di reset aggiuntive
